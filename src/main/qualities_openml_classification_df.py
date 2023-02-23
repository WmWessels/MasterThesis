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
    indexes = [1241,
 1242,
 1575,
 4102,
 40945,
 41496,
 41701,
 41703,
 41705,
 42252,
 42813,
 43051,
 43160,
 43890,
 43891,
 43892,
 43896,
 350,
 351,
 357,
 373,
 374,
 376,
 379,
 380,
 43897,
 43903,
 43904,
 43905,
 44073,
 44074,
 44075,
 44076,
 44078,
 44079,
 44080,
 44082,
 44083,
 44084,
 44085,
 44086,
 44087,
 44088,
 44089,
 44090,
 44091,
 44093,
 44094,
 44095,
 44096,
 44120,
 44122,
 44123,
 44124,
 44125,
 44126,
 44127,
 44128,
 44130,
 44131,
 44154,
 44155,
 44156,
 44157,
 44158,
 44160,
 44161,
 44162,
 44185,
 44186,
 44237,
 44238,
 44239,
 44240,
 44241,
 44242,
 44243,
 44244,
 44245,
 44246,
 44247,
 44248,
 44249,
 44250,
 44251,
 44252,
 44271,
 44272,
 44273,
 44274,
 44275,
 44276,
 44277,
 44278,
 44279,
 44280,
 44281,
 44282,
 44283,
 44284,
 44285,
 44286,
 44287,
 44288,
 44289,
 44290,
 44291,
 44292,
 44293,
 44294,
 44295,
 44296,
 44297,
 44298,
 44299,
 44300,
 44301,
 44302,
 44303,
 44304,
 44305,
 44306,
 44307,
 44308,
 44309,
 44310,
 44312,
 44313,
 44314,
 44315,
 44316,
 44317,
 44318,
 44319,
 44320,
 44321,
 44322,
 44323,
 44324,
 44325,
 44326,
 44327,
 44328,
 44329,
 44330,
 44331,
 44332,
 44333,
 44334,
 44335,
 44336,
 44337,
 44338,
 44340,
 44341,
 44342,
 44343,
 44344,
 45019,
 45020,
 45021,
 45022,
 45026,
 45028,
 45035,
 45036,
 45037,
 45038,
 45039,
 45049,
 45058,
 45059,
 45060,
 45061,
 45062,
 45063,
 45064,
 45065,
 45066,
 45067,
 45068,
 45069,
 45070,
 45073,
 45074,
 45075,
 45076,
 45077,
 45078,
 45079,
 389]
    ftrs = list(mapping.keys())
    
    counter = 0
    lm_features = ["best_node", "linear_discr", "naive_bayes", "random_node", "worst_node"]
    mfe_landmark = MFE(features = lm_features, groups = ["landmarking"], num_cv_folds = 5, lm_sample_frac = 0.5, measure_time = "total", suppress_warnings=True)
    mfe = MFE(features = ftrs, groups = ["general", "statistical", "info-theory"], summary = ["nanmean", "nansd"], measure_time = "total", suppress_warnings=True)
    columns = list(mfe.extract_metafeature_names())
    columns_lm = list(mfe_landmark.extract_metafeature_names())
    columns.extend(columns_lm)
    columns.extend(["missing.mean", "missing.sd"])
    meta_features = []
    times = []
    total_times = []
  
        
    for idx in indexes:

        print(idx)
        try:
            dataset = datasets.get_dataset(idx)
            X, y, cat_mask, attr = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)
            cat_cols = [b for a,b in zip(cat_mask, attr) if a]
            X, y = X.to_numpy(), y.to_numpy()
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
            mfe_landmark.fit(imputed_X, y, cat_cols = cat_cols)
            mfe.fit(X, y, cat_cols = cat_cols)
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
    df = pd.DataFrame(index = list(indexes), data = meta_features, columns = columns)
    time_df = pd.DataFrame(index = list(indexes), data = times, columns = columns)
    total_time_df = pd.DataFrame(index = list(indexes), data = total_times)
    df.to_csv("../data/final/additional_parses.csv")
    time_df.to_csv("../data/times_final/additional_parses.csv")
    total_time_df.to_csv("../data/total_times/additional_parses.csv")


if __name__ == "__main__":
    main()