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
                    "freq_class": "get_freq_class",
                    "nr_attr": "get_nr_attr",
                    "nr_bin": "get_nr_bin",
                    "nr_cat": "get_nr_cat",
                    "nr_class": "get_nr_class",
                    "nr_inst": "get_nr_inst",
                    "nr_num": "get_nr_num"
}

mapping_stat = {
                    "cor": "get_cor",
                    "cov": "get_cov",
                    "iq_range": "get_iqr",
                    "kurtosis": "get_kurtosis",
                    "max": "get_max",
                    "mean": "get_mean",
                    "median": "get_median",
                    "min": "get_min",
                    "nr_outliers": "get_outliers",
                    "sd": "get_std",
                    "skewness": "get_skewness",
                    "var": "get_var",
                    #sparsity?
}

mapping_infotheory = {
                    "attr_ent": ["ft_attr_ent", "C"],
                    "class_conc": ["ft_class_conc", "C", "y"],
                    "eq_num_attr": ["ft_eq_num_attr", "C", "y"],
                    "joint_ent": ["ft_joint_ent", "C", "y"],
                    "attr_conc": ["ft_attr_conc", "C"]
}