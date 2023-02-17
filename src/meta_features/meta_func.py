import numpy as np
import scipy
import pandas as pd

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
    def get_freq_class(y: np.array):
        _, counts = np.unique(y, return_counts=True)
        freq = counts / y.size
        return freq.nanmean(), freq.std()

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
    def get_iqr(X: np.array):
        iqr = scipy.stats.iqr(X, axis = 0, nan_policy = "omit")
        return np.nanmean(iqr), np.nanstd(iqr)
    @staticmethod
    def get_cor(X: np.array):
        df = pd.DataFrame(X)
        cor = df.corr().values
        return np.nanmean(cor), np.nanstd(cor)
    
    @staticmethod
    def get_cov(X: np.array):
        df = pd.DataFrame(X)
        cov = df.cov().values
        return np.nanmean(cov), np.nanstd(cov)
    @staticmethod
    def get_kurtosis(X):
        df = pd.DataFrame(X)
        kurtosis = df.kurtosis(axis = 0).values
        return np.nanmean(kurtosis), np.nanstd(kurtosis)
    @staticmethod
    def get_max(X):
        df = pd.DataFrame(X)
        max_ = df.max(axis = 0).values
        return np.nanmean(max_), np.nanstd(max_)
    @staticmethod
    def get_mean(X):
        df = pd.DataFrame(X)
        mean_ = df.mean(axis = 0).values
        return np.nanmean(mean_), np.nanstd(mean_)
    @staticmethod
    def get_median(X):
        df = pd.DataFrame(X)
        median_ = df.median(axis = 0).values
        return np.nanmean(median_), np.nanstd(median_)
    @staticmethod
    def get_min(X):
        df = pd.DataFrame(X)
        min_ = df.min(axis = 0).values
        return np.nanmean(min_), np.nanstd(min_)
    @staticmethod
    def get_std(X):
        df = pd.DataFrame(X)
        std = df.std(axis = 0).values
        return np.nanmean(std), np.nanstd(std)
    @staticmethod
    def get_skewness(X):
        df = pd.DataFrame(X)
        skew = df.skew(axis = 0).values
        return np.nanmean(skew), np.nanmean(skew)
    @staticmethod
    def get_var(X):
        df = pd.DataFrame(X)
        var = df.var(axis = 0).values
        return np.nanmean(var), np.nanstd(var)
    @staticmethod
    def get_outliers(X, whis: float = 1.5):
        if type(X) == pd.DataFrame:
            X = X.values
        v_min, q_1, q_3, v_max = np.percentile(X, (0, 25, 75, 100), axis=0)

        whis_iqr = whis * (q_3 - q_1)

        cut_low = q_1 - whis_iqr
        cut_high = q_3 + whis_iqr

        return np.sum(np.logical_or(cut_low > v_min, cut_high < v_max))
