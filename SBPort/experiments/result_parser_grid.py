import argparse
import os
import numpy as np
from itertools import product
import shutil
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler

CLUSTER_OPTIONS = [3, 5, 8, "optics"]
PORTFOLIO_SIZES = [4, 8, 16]
np.set_printoptions(suppress=True)

cluster_port_combinations = list(product(PORTFOLIO_SIZES, CLUSTER_OPTIONS))

def evaluate_validation_results(path: Path, inverse = False):
    results = {}
    with open(path, "r") as f:
        data = f.readlines()
    data = [line.replace("nan", "np.nan") for line in data]
    
    for key in CLUSTER_OPTIONS:
        arr = np.array([eval(line.split(f"{key}:")[-1]) for line in data if f"result on {key}" in line])
        arr = np.where(arr == 0.5, np.nan, arr)
        arr = np.abs(arr) if inverse else arr
        results[str(key)] = arr
    # arr = [eval(line.split("result on optics:")[-1]) for line in data if "result on optics:" in line]
    # # arr = [eval(line.split("result:")[-1]) for line in data if "result:" in line]
    # arr = np.where(arr == 0.5, np.nan, arr)
    # arr = np.abs(arr) if inverse else arr
    # results["optics"] = arr

    return results

def best_performance(
    results: dict,
    greater_is_better: bool = False
):
    """
    Decide on the best performing policy by looking at the best performing portfolio for every combination of portfolio size and cluster option.
    Ties are broken by looking at the amount of missing values in the portfolio on a per data set basis.
    """
    items = []
    for portfolio_size in PORTFOLIO_SIZES:
        for key in CLUSTER_OPTIONS:
            current_arr = results[str(key)][:, :portfolio_size]
            my_list = [np.around(np.nanmax(result), 3) for result in current_arr] if greater_is_better else [np.around(np.nanmin(result), 3) for result in current_arr]
            items.append(my_list)
    result = np.array(items).T
    print(result)
    max_value_per_dataset = np.nanmax(result, axis = 1) if greater_is_better else np.nanmin(result, axis = 1)
    #count the number of elements that is equal to the max value per configuration
    amount_of_times_equal_to_max = np.sum(result == max_value_per_dataset[:, None], axis = 0)
    ind = np.nanargmax(amount_of_times_equal_to_max)

    #Below code is for breaking ties

    unique_elements, counts = np.unique(ind, return_counts = True)

    element_with_max_count = unique_elements[np.where(counts == np.max(counts))]

    if len(element_with_max_count) > 1:
        missing_count = np.isnan(result).sum(axis = 0)
        min_missing = np.argmax(missing_count[element_with_max_count])
        return cluster_port_combinations[element_with_max_count[min_missing]]

    return cluster_port_combinations[element_with_max_count[0]]


def heuristic_metric(
    results: dict,
    greater_is_better: bool = False,
    scale = False
):
    """
    Heuristic Metric is defined as follows:
    For every combination of portfolio size and cluster option, we apply the following formula:
    (average of the performance of the portfolio +- amount of missing values in the portfolio) / 2, 
    where the + or - is dependent on the greater_is_better parameter being False or True, respectively.

    Note that we scale the performance values to be between 0 and 1 on a per dataset basis, 
    such that we can aggregate the results over the same scale.
    
    """
    items = []
    missings = []
    if scale:
        arr = np.concatenate(tuple(results[str(cluster_option)] for cluster_option in CLUSTER_OPTIONS), axis = 1).T
        # Scale using robust scaler, such that we do not overly penalize outliers while aggregating the results
        robust_scaler = RobustScaler()
        scaler = robust_scaler.fit(arr)
        for key in CLUSTER_OPTIONS:
            current_arr = results[str(key)]
            results[str(key)] = scaler.transform(current_arr.T).T

    for portfolio_size in PORTFOLIO_SIZES:
        for key in CLUSTER_OPTIONS:
            # Slice the array for the current portfolio size
            current_arr = results[str(key)][:, :portfolio_size]
            amount_missing = np.sum(np.isnan(current_arr), axis = 1)/portfolio_size
            missings.append(amount_missing)

            # Get the best performing value from the array, calculate the difference to the best value for every element on a per dataset basis
            best_per_dataset = np.nanmax(current_arr, axis = 1) if greater_is_better else np.nanmin(current_arr, axis = 1)

            difference_to_best = np.array(np.abs(current_arr - best_per_dataset[:, np.newaxis]))
            difference_to_best = np.subtract(1, difference_to_best) if greater_is_better else difference_to_best
            if scale:
                difference_to_best = MinMaxScaler().fit_transform(difference_to_best)
            avg_per_dataset = np.nanmean(difference_to_best, axis = 1)

            items.append(avg_per_dataset)

    if greater_is_better:
        heuristic_score = np.around(np.maximum((np.array(items)  + (1 - np.array(missings)) )/2, 0), 3)
        heuristic_score = np.where(np.isnan(heuristic_score), 0, heuristic_score)
        #for every row, get the indices where the value is equal to the max_
        mean_perf = np.mean(heuristic_score, axis = 1)[-len(CLUSTER_OPTIONS):]
        ind = np.argmax(mean_perf)
    else:
        heuristic_score = np.around((np.array(items) + np.array(missings))/2, 3)
        heuristic_score = np.where(np.isnan(heuristic_score), 1, heuristic_score)
        mean_perf = np.mean(heuristic_score, axis = 1)
        ind = np.argmin(mean_perf[-len(CLUSTER_OPTIONS):])
    print(heuristic_score.T)

    return cluster_port_combinations[ind+8]

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"], required = True)
    args = argparser.parse_args()
    task = args.task

    if task == "bin":
        results = evaluate_validation_results(Path(__file__).parent.parent / "results" / "grid_search" / "bin_val.out")
        result_max = best_performance(results, greater_is_better=True)
        result_heuristic = heuristic_metric(results, greater_is_better = True)
        
    elif task == "multi":
        results = evaluate_validation_results(Path(__file__).parent.parent / "results" / "grid_search" / "multi_val.out", inverse=True)
        result_max = best_performance(results, greater_is_better=False)
        result_heuristic = heuristic_metric(results, greater_is_better = False, scale = True)
        
    elif task == "regr":
        results = evaluate_validation_results(Path(__file__).parent.parent / "results" / "grid_search" / "regr_val.out")
        result_max = best_performance(results, greater_is_better=False)
        result_heuristic = heuristic_metric(results, greater_is_better = False, scale = True)
          
    print("heuristic policy: ", result_heuristic)
    print("max policy: ", result_max)

    if "optics" == result_heuristic[1]:
        heuristic_prefix = f"optics_psize_{result_heuristic[0]}"
    else:
        heuristic_prefix = f"kernel_kmeans_{result_heuristic[1]}_psize_{result_heuristic[0]}"

    if "optics" == result_max[1]:
        max_prefix = f"optics_psize_{result_max[0]}"
    else:
        max_prefix = f"kernel_kmeans_{result_max[1]}_psize_{result_max[0]}"
    
    if not os.path.isdir(Path(__file__).parent.parent / "optimal_configurations" / f"{task}"):
        os.mkdir(Path(__file__).parent.parent / "optimal_configurations" / f"{task}")

    shutil.copy(
        Path(__file__).parent.parent / "inference_pipelines" / f"{task}" / f"inference_pipeline_{task}_{heuristic_prefix}", 
        Path(__file__).parent.parent / "optimal_configurations" / f"{task}" / f"optimal_{task}_heuristic_{result_heuristic[1]}_psize_{result_heuristic[0]}"
    )

    shutil.copy(
        Path(__file__).parent.parent / "inference_pipelines" / f"{task}" / f"inference_pipeline_{task}_{max_prefix}",
        Path(__file__).parent.parent / "optimal_configurations" / f"{task}" / f"optimal_{task}_max_{result_max[1]}_psize_{result_max[0]}"
    )

