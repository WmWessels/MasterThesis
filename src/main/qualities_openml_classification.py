import openml
from openml.tasks import TaskType
import pandas as pd
import numpy as np

from pymfe.mfe import MFE
from metafeature_mapping import mapping_general, mapping_infotheory, mapping_stat
from utils import batch
from sklearn.impute import SimpleImputer

from gama.utilities.preprocessing import basic_encoding

def main() -> None:
    import time

    datasets = openml.datasets
    classification_datasets = openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe")
    large_indexes = set(datasets.list_datasets(output_format="dataframe").query("NumberOfInstances > 500000 or NumberOfFeatures > 2500").index)
    
    all_indexes = set(classification_datasets.groupby('did').first().index)
    large_sizes = set([  273,   274,   554,  1111,  1112,  1113,  1114,  1581,  1582,
        1583,  1584,  1585,  1586,  1587,  1588,  1592,  4133, 4140,  4535,
        6331, 23513, 40517, 40923, 40996, 41039, 41065, 41138, 41147,
       41163, 41166, 41167, 41982, 41986, 41988, 41990, 41991, 42108, 42260,
       42343, 42395, 42396, 42435, 42571, 42705, 42737, 42803, 42973,
       43069, 43305, 43306, 43357, 43428, 43430, 43496, 43764, 43846,
       43948, 44036, 44045, 44060, 44067, 44159, 45044, 45046, 45071,
       45082])

    bad_datasets = set([914, 919, 924, 4133, 42260, 42491])
    filtered_indexes = sorted(list(all_indexes - large_indexes - large_sizes - bad_datasets))
    mapping = mapping_general | mapping_stat | mapping_infotheory

    ftrs = list(mapping.keys())
    
    counter = 0
    lm_features = ["best_node", "linear_discr", "naive_bayes", "random_node", "worst_node"]
    mfe_landmark = MFE(features = lm_features, groups = ["landmarking"], num_cv_folds = 5, lm_sample_frac = 0.5, measure_time = "total", suppress_warnings=True)
    mfe = MFE(features = ftrs, groups = ["general", "statistical", "info-theory"], summary = ["nanmean", "nansd"], measure_time = "total", suppress_warnings=True)
    columns = list(mfe.extract_metafeature_names())
    columns_lm = list(mfe_landmark.extract_metafeature_names())
    columns.extend(columns_lm)
    
    for batch_idx in batch(filtered_indexes, 20):
        meta_features = []
        times = []
        total_times = []
        for idx in batch_idx:

            print(idx)
            try:
                dataset = datasets.get_dataset(idx)
                X, y, cat_mask, _ = dataset.get_data(dataset_format="array", target = dataset.default_target_attribute)
                begin_time = time.time()
                num_mask = [not elem for elem in cat_mask]
                C = X[:, cat_mask]
                N = X[:, num_mask]
                imputer_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
                if len(C[0, :]) > 0:
                    C = imputer_cat.fit_transform(C)
                imputer_num = SimpleImputer(strategy='mean', missing_values=np.nan)
                if len(N[0, :]) > 0:
                    N = imputer_num.fit_transform(N)
                imputed_X = np.concatenate((C, N), axis = 1)
                end_time = time.time()
                print(f"finished preprocessing in {end_time - begin_time}")
                time_calc_begin = time.time()
                if y is None:
                    y = X[:, -1]
                    imputed_X = imputed_X[:, :-1]
                    X = X[:, :-1]
                    print("y was changed", y)
                mfe_landmark.fit(imputed_X, y)
                mfe.fit(X, y)
                _, vals, time_mf = mfe.extract()
                _, vals_lm, time_lm = mfe_landmark.extract()
                time_calc_end = time.time()
                total_time = time_calc_end - time_calc_begin
                total_times.append(total_time)
                time_mf.extend(time_lm)
                times.append(time_mf)
                vals.extend(vals_lm)
                meta_features.append(vals)
                
            except Exception as e:
                print(e)
                meta_features.append([None for i in range(len(columns))])
                times.append([None for i in range(len(columns))])
                total_times.append(None)
        df = pd.DataFrame(index = list(batch_idx), data = meta_features, columns = columns)
        time_df = pd.DataFrame(index = list(batch_idx), data = times, columns = columns)
        total_time_df = pd.DataFrame(index = list(batch_idx), data = total_times)
        df.to_csv(f"../data/final/meta_features_classification_{counter}.csv")
        time_df.to_csv(f"../data/times_final/meta_features_classification_{counter}.csv")
        total_time_df.to_csv(f"../data/total_times/total_times_{counter}.csv")
        
        counter += 1

if __name__ == "__main__":
    main()