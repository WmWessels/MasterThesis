import openml
from openml.tasks import TaskType
import pandas as pd
from pymfe.mfe import MFE

def main() -> None:

    datasets = openml.datasets
    classification_datasets = openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe")
    large_indexes = set(datasets.list_datasets(output_format="dataframe").query(("NumberOfInstances > 500000")).index)
    
    all_indexes = set(classification_datasets.groupby('did').first().index)
    all_indexes.remove([273, 274])
    filtered_indexes = sorted(list(all_indexes - large_indexes))
    print(filtered_indexes)
    mfe = MFE(groups=["general", "statistical", "info-theory", "model-based"], suppress_warnings=True)
    meta_features = []
    for idx in filtered_indexes:
        print(idx)
        try:
            dataset = datasets.get_dataset(idx)
            X, y, _, _ = dataset.get_data(dataset_format="array", target = dataset.default_target_attribute)
            mfe.fit(X, y)
            columns, values = mfe.extract()
            meta_features.append(values)
            # meta_features.append(meta_features_temp)
        except:
            meta_features.append({})

    pd.DataFrame(index = all_indexes, data = meta_features, columns = columns).to_csv(f"meta_features_classification.csv")

if __name__ == "__main__":
    main()