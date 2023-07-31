import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_results_over_time(
    path: Path
):
    data = pd.read_csv(path)
    data_selected = data[["t_wallclock", "score"]]
    data_selected = data_selected.sort_values(by = "t_wallclock")
    data_selected["score"] = data_selected["score"].apply(lambda x: eval(x.replace("-inf", "np.nan"))[0])

