import openml
from openml.tasks import TaskType
import pandas as pd

def main() -> None:
    datasets = openml.datasets
    classification_datasets = openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_REGRESSION, output_format="dataframe")
    all_indexes = classification_datasets.groupby('did').first().index
    
    meta_features = []
    for idx in all_indexes:
        try:
            curr_data = datasets.get_dataset(idx, download_data = False)
            meta_features_temp = curr_data.qualities
            meta_features.append(meta_features_temp)
        except:
            meta_features.append({})

    pd.DataFrame(index = all_indexes, data = meta_features).to_csv(f"meta_features_regression.csv")

if __name__ == "__main__":
    main()