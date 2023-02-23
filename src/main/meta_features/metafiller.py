import numpy as np
import scipy
import pandas as pd
from pymfe.info_theory import MFEInfoTheory
from pymfe.landmarking import MFELandmarking
from sklearn.metrics import accuracy_score
from typing import Optional, Callable


class BackFiller:
    
    @staticmethod
    def get_missing(X: np.array):
        missing_per_column = np.count_nonzero(np.isnan(X), axis = 0)/X.shape[0]
        return np.mean(missing_per_column), np.std(missing_per_column)

    @staticmethod
    def get_nr_attr(X: np.array):
        return X.shape[1]
    
    @staticmethod
    def get_nr_inst(X: np.array):
        return X.shape[0]

    @staticmethod
    def get_nr_bin(X: np.array):
        """
        From pymfe
        """
        bin_cols = np.apply_along_axis(
                func1d=lambda col: np.unique(col).size == 2, axis=0, arr=X
            )

        return np.sum(bin_cols)
    
    @staticmethod
    def get_nr_cat(X: np.array):
        df = pd.DataFrame(X)
        return df.select_dtypes(include=["category"]).shape[1]

    @staticmethod
    def get_nr_class(y: np.array):
        return len(np.unique(y))

    @staticmethod
    def get_freq_class_mean(y: np.array):
        _, counts = np.unique(y, return_counts=True)
        freq = counts / y.size
        return freq.nanmean()

    @staticmethod
    def get_freq_class_sd(y: np.array):
        _, counts = np.unique(y, return_counts=True)
        freq = counts / y.size
        return freq.std()

    @staticmethod
    def get_nr_num(X: np.array):
        df = pd.DataFrame(X)
        return df.select_dtypes(np.number).shape[1]
    
    def get_num_to_cat(self, X):
        nr_num = self.get_nr_num(X)
        nr_cat = self.get_nr_cat(X)
        return 0 if nr_cat == 0 else nr_num / nr_cat

    @staticmethod
    def get_attr_to_inst(nr_inst: int, nr_attr: int):
        return nr_attr / nr_inst

    @staticmethod
    def get_iqr_mean(X: np.array):
        iqr = scipy.stats.iqr(X, axis = 0, nan_policy = "omit")
        return np.nanmean(iqr)

    @staticmethod
    def get_iqr_sd(X: np.array):
        iqr = scipy.stats.iqr(X, axis = 0, nan_policy = "omit")
        return np.nanstd(iqr)

    @staticmethod
    def get_cor_mean(X: np.array):
        df = pd.DataFrame(X)
        cor = df.corr().values
        return np.nanmean(cor)

    @staticmethod
    def get_cor_sd(X: np.array):
        df = pd.DataFrame(X)
        cor = df.corr().values
        return np.nanstd(cor)
    
    @staticmethod
    def get_cov_mean(X: np.array):
        df = pd.DataFrame(X)
        cov = df.cov().values
        return np.nanmean(cov)

    @staticmethod
    def get_cov_sd(X: np.array):
        df = pd.DataFrame(X)
        cov = df.cov().values
        return np.nanstd(cov)

    @staticmethod
    def get_kurtosis_mean(X):
        df = pd.DataFrame(X)
        kurtosis = df.kurtosis(axis = 0).values
        return np.nanmean(kurtosis), np.nanstd(kurtosis)
    
    @staticmethod
    def get_kurtosis_sd(X):
        df = pd.DataFrame(X)
        kurtosis = df.kurtosis(axis = 0).values
        return np.nanstd(kurtosis)

    @staticmethod
    def get_max_mean(X):
        df = pd.DataFrame(X)
        max_ = df.max(axis = 0).values
        return np.nanmean(max_)

    @staticmethod
    def get_max_sd(X):
        df = pd.DataFrame(X)
        max_ = df.max(axis = 0).values
        return np.nanstd(max_)

    @staticmethod
    def get_mean_mean(X):
        df = pd.DataFrame(X)
        mean_ = df.mean(axis = 0).values
        return np.nanmean(mean_)
    
    @staticmethod
    def get_mean_sd(X):
        df = pd.DataFrame(X)
        mean_ = df.mean(axis = 0).values
        return np.nanstd(mean_)

    @staticmethod
    def get_median_mean(X):
        df = pd.DataFrame(X)
        median_ = df.median(axis = 0).values
        return np.nanmean(median_)
    
    @staticmethod
    def get_median_sd(X):
        df = pd.DataFrame(X)
        median_ = df.median(axis = 0).values
        return np.nanstd(median_)

    @staticmethod
    def get_min_mean(X):
        df = pd.DataFrame(X)
        min_ = df.min(axis = 0).values
        return np.nanmean(min_)

    @staticmethod
    def get_min_sd(X):
        df = pd.DataFrame(X)
        min_ = df.min(axis = 0).values
        return np.nanstd(min_)

    @staticmethod
    def get_std_mean(X):
        df = pd.DataFrame(X)
        std = df.std(axis = 0).values
        return np.nanmean(std)
    
    @staticmethod
    def get_std_sd(X):
        df = pd.DataFrame(X)
        std = df.std(axis = 0).values
        return np.nanstd(std)

    @staticmethod
    def get_skewness_mean(X):
        df = pd.DataFrame(X)
        skew = df.skew(axis = 0).values
        return np.nanmean(skew)
    
    @staticmethod
    def get_skewness_sd(X):
        df = pd.DataFrame(X)
        skew = df.skew(axis = 0).values
        return np.nanstd(skew)

    @staticmethod
    def get_var_mean(X):
        df = pd.DataFrame(X)
        var = df.var(axis = 0).values
        return np.nanmean(var)
    
    @staticmethod
    def get_var_sd(X):
        df = pd.DataFrame(X)
        var = df.var(axis = 0).values
        return np.nanstd(var)
        
    @staticmethod
    def get_outliers(X, whis: float = 1.5):
        if type(X) == pd.DataFrame:
            X = X.values
        v_min, q_1, q_3, v_max = np.percentile(X, (0, 25, 75, 100), axis=0)

        whis_iqr = whis * (q_3 - q_1)

        cut_low = q_1 - whis_iqr
        cut_high = q_3 + whis_iqr

        return np.sum(np.logical_or(cut_low > v_min, cut_high < v_max))

    @staticmethod
    def get_attr_ent_mean(X: np.array, y: Optional[np.array] = None):
        attr_conc = MFEInfoTheory.ft_attr_conc(X)
        return np.nanmean(attr_conc)

    @staticmethod
    def get_attr_ent_sd(X: np.array, y: Optional[np.array] = None):
        attr_conc = MFEInfoTheory.ft_attr_conc(X)
        return np.nanstd(attr_conc)

    @staticmethod
    def get_class_conc_mean(X: np.array, y: np.array):
        class_conc = MFEInfoTheory.ft_class_conc(X, y)
        return np.nanmean(class_conc)
    
    @staticmethod
    def get_class_conc_sd(X: np.array, y: np.array):
        class_conc = MFEInfoTheory.ft_class_conc(X, y)
        return np.nanstd(class_conc)
    
    @staticmethod
    def get_eq_num_attr_mean(X: np.array, y: np.array):
        eq_num_attr = MFEInfoTheory.ft_eq_num_attr(X, y)
        return np.nanmean(eq_num_attr)

    @staticmethod
    def get_eq_num_attr_sd(X: np.array, y: np.array):
        eq_num_attr = MFEInfoTheory.ft_eq_num_attr(X, y)
        return np.nanmean(eq_num_attr)
    
    @staticmethod
    def get_joint_ent_mean(X: np.array, y: np.array):
        joint_ent = MFEInfoTheory.ft_eq_num_attr(X, y)
        return np.nanmean(joint_ent)

    @staticmethod
    def get_joint_ent_sd(X: np.array, y: np.array):
        joint_ent = MFEInfoTheory.ft_eq_num_attr(X, y)
        return np.nanstd(joint_ent)

    @staticmethod
    def get_attr_conc_mean(X: np.array, y: Optional[np.array] = None):
        attr_conc = MFEInfoTheory.ft_attr_conc(X)
        return np.nanmean(attr_conc)

    @staticmethod
    def get_attr_conc_sd(X: np.array, y: Optional[np.array] = None):
        attr_conc = MFEInfoTheory.ft_attr_conc(X)
        return np.nanstd(attr_conc)