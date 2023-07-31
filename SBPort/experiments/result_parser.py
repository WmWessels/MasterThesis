import numpy as np 
from pathlib import Path
from typing import Optional
import argparse
from sklearn.preprocessing import RobustScaler, MinMaxScaler

np.set_printoptions(suppress=True)

def calculate_heuristic_policy_value(
    results: list,
    scale: bool = False
):
    amount_missing = np.sum(np.isnan(results))/len(results)
    if scale:

        best_result = np.nanmax(results)
        distances_to_best = np.subtract(best_result, results)
        distances_to_best = np.abs(distances_to_best)
        # min max scaler scales from 0 to 1, so we need to subtract from 1, as our best result is 0
        results = np.subtract(1, MinMaxScaler().fit_transform(distances_to_best.reshape(-1, 1)))

        return np.around((np.array(np.nanmean(results)) + amount_missing)/2, 3)
    return np.around((np.nanmean(results) + (1 - amount_missing))/2, 3)

def parse_portfolio_results_max(
    path: Path
):
    with open(path, "r") as f:
        data = f.readlines()
    
    data = [line.replace("nan", "np.nan") for line in data]
    lines = []

    for line in data:
        if "result on" in line:
            lines.append(eval(line.split(":")[-1]))
    
    lines = np.where(lines == 0.5, np.nan, lines)
    np.around(lines, decimals = 3, out = lines)

    return np.nanmax(lines, axis = 1)
    
    # return heuristic policy values

def parse_portfolio_results_heuristic(
    path: Path,
    portfolio_size: Optional[int] = None,
    scale: bool = False
):
    with open(path, "r") as f:
        data = f.readlines()
    
    data = [line.replace("nan", "np.nan") for line in data]
    lines = []

    for line in data:
        if "result on" in line:
            lines.append(eval(line.split(":")[-1]))
    
    lines = np.where(lines == 0.5, np.nan, lines)
    np.around(lines, decimals = 3, out = lines)
    if not portfolio_size:
        return np.array([calculate_heuristic_policy_value(result, scale = scale) for result in lines])
    return np.array([calculate_heuristic_policy_value(result, scale = scale) for result in lines[:, :portfolio_size]])


def parse_clustervsall_results(
    path: Path,
    cluster_size: int
):
    with open(path, "r") as f:
        data = f.readlines()

    true_labels = []
    
    for line in data:
        if "predicted_label" in line:
            true_labels.append(eval(line.split(":")[-1])[0])
    
    results = []
    for line in data:
        if "result on" in line:
            temp_line = line.replace("nan", "np.nan")
            results.append(eval(temp_line.split(":")[-1]))
    
    results_dict = {
        str(dataset_id): results[i: i + cluster_size] for dataset_id, i in zip(range(len(results)), range(0, len(results), cluster_size))
    }
    aggregated_results = []
    predicted_values = []
    for label, (_, value) in zip(true_labels, results_dict.items()):
        predicted_value = value[label]
        predicted_values.append(np.nanmax(predicted_value))
        result_per_dataset = []
        for item in value:
            if not item == predicted_value:
                result_per_dataset.append(np.nanmax(item))
        aggregated_results.append(result_per_dataset)
    for true, other in zip(predicted_values, aggregated_results):
        print(true, other)

    # for label, (_, value) in zip(true_labels, results_dict.items()):
    #     predicted_value = value[label]
    #     predicted_values.append(calculate_heuristic_policy_value(predicted_value, scale = True))
    #     result_per_dataset = []
    #     for item in value:
    #         if not item == predicted_value:
    #             result_per_dataset.append(calculate_heuristic_policy_value(item, scale = True))
    #     aggregated_results.append(result_per_dataset)
    # for true, other in zip(predicted_values, aggregated_results):
    #     print(true, np.mean(other))

# parse_clustervsall_results(Path(__file__).parent.parent / "results" / "clustervsall" / "clustervsall_regr.out", cluster_size = 3)
# results_one_nn = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "one_nn" / "one_nn_multi.out", portfolio_size = 4, scale = True)
# results_static = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "static" / "static_multi.out", portfolio_size = 4, scale = True)
# results_opt = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "optimal" / "optimal_multi_heur.out", portfolio_size = 4, scale = True)
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"], required = True)
    args = argparser.parse_args()
    task = args.task
    if task in ["bin", "multi"]:
        cluster_size = 8
    else:
        cluster_size = 3
    parse_clustervsall_results(Path(__file__).parent.parent / "results" / "clustervsall" / f"clustervsall_{task}.out", cluster_size = cluster_size)

    path_one_nn = Path(__file__).parent.parent / "results" / "one_nn" / f"one_nn_{task}.out"
    results_one_nn = parse_portfolio_results_max(path_one_nn)
    path_static = Path(__file__).parent.parent / "results" / "static" / f"static_{task}.out"
    results_static = parse_portfolio_results_max(path_static)
    path_opt = Path(__file__).parent.parent / "results" / "optimal" / f"optimal_{task}.out"
    results_opt = parse_portfolio_results_max(path_opt)
    print("MAX POLICY")
    print("one_nn, static, optimal")
    print(np.array([results_one_nn, results_static, results_opt]).T)
    if task in ["multi", "regr"]:
        results_one_nn = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "one_nn" / f"one_nn_{task}.out", scale = True)
        results_static = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "static" / f"static_{task}.out", scale = True)
        results_opt = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "optimal" / f"optimal_{task}_heur.out", scale = True)
    else:
        results_one_nn = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "one_nn" / f"one_nn_{task}.out", scale = False)
        results_static = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "static" / f"static_{task}.out", scale = False)
        results_opt = parse_portfolio_results_heuristic(Path(__file__).parent.parent / "results" / "optimal" / f"optimal_{task}_heur.out", scale = False)
    print("HEURISTIC POLICY")
    print("one_nn, static, optimal")
    print(np.array([results_one_nn, results_static, results_opt]).T)

