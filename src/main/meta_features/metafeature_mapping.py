# mapping_general = {
#                     "num_to_cat": "get_num_to_cal",
#                     "freq_class": "get_freq_class",
#                     "nr_attr": "get_nr_attr",
#                     "nr_bin": "get_nr_bin",
#                     "nr_cat": "get_nr_cat",
#                     "nr_class": "get_nr_class",
#                     "nr_inst": "get_nr_inst",
#                     "nr_num": "get_nr_num"
# }

mapping_general = {
                    "num_to_cat": "get_num_to_cat",
                    "freq_class.nanmean": "get_freq_class_mean",
                    "freq_class.nansd": "get_freq_class_sd",
                    "nr_attr": "get_nr_attr",
                    "nr_bin": "get_nr_bin",
                    "nr_cat": "get_nr_cat",
                    "nr_class": "get_nr_class",
                    "nr_inst": "get_nr_inst",
                    "nr_num": "get_nr_num"
}

mapping_stat = {
                    "cor.nanmean": "get_cor_mean",
                    "cor.nansd": "get_cor_sd",
                    "cov.nanmean": "get_cov_mean",
                    "cov.nansd": "get_cov.sd",
                    "iq_range.nanmean": "get_iqr_mean",
                    "iq_range.nansd": "get_iqr_sd",
                    "kurtosis.nanmean": "get_kurtosis_mean",
                    "kurtosis.nansd": "get_kurtosis_sd",
                    "max.nanmean": "get_max_mean",
                    "max.nansd": "get_max_sd",
                    "mean.nanmean": "get_mean_mean",
                    "mean.nansd": "get_mean_sd",
                    "median.nanmean": "get_median_mean",
                    "median.nansd": "get_median_sd",
                    "min.nanmean": "get_min_mean",
                    "min.nansd": "get_min_sd",
                    "nr_outliers": "get_outliers",
                    "sd.nanmean": "get_std_mean",
                    "sd.nansd": "get_std_sd",
                    "skewness.nanmean": "get_skewness_mean",
                    "skewness.nansd": "get_skewness_sd",
                    "var.nanmean": "get_var_mean",
                    "var.nansd": "get_var_sd"
                    #sparsity?
}

mapping_infotheory = {
                    "attr_ent": ["ft_attr_ent", "C"],
                    "class_conc": ["ft_class_conc", "C", "y"],
                    "eq_num_attr": ["ft_eq_num_attr", "C", "y"],
                    "joint_ent": ["ft_joint_ent", "C", "y"],
                    "attr_conc": ["ft_attr_conc", "C"]
}

mapping_landmarking = {
                    "best_node": "0.75",
                    "linear_discr": "0.75",
                    "naive_bayes": "0.75",
                    "random_node": "0.75",
                    "worst_node": "0.75"
}

mapping = mapping_general | mapping_stat | mapping_infotheory | mapping_landmarking