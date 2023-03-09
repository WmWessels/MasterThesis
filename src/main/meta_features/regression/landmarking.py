"""Regression variant of Pymfe's landmarking module."""

import typing as t

import numpy as np
import sklearn.model_selection
import sklearn.naive_bayes
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import r2_score


class LandmarkingRegressor:
    

    @classmethod
    def precompute_landmarking_sample(
        cls,
        N: np.ndarray,
        lm_sample_frac: float,
        random_state: t.Optional[int] = None,
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute subsampling landmarking subsample indices.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        lm_sample_frac : float
            The percentage of examples subsampled. Value different from default
            will generate the subsampling-based relative landmarking
            metafeatures.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``sample_inds`` (:obj:`np.ndarray`): indices related to the
                  subsampling of the original dataset. Used only if the
                  subsampling landmarking method is used and, therefore, this
                  value is only precomputed if ``lm_sample_frac`` is less
                  than 1.0.
        """
        precomp_vals = {}

        if N is not None and N.size > 0 and "sample_inds" not in kwargs:
            if lm_sample_frac < 1.0:
                num_inst, _ = N.shape

                precomp_vals["sample_inds"] = cls._get_sample_inds(
                    num_inst=num_inst,
                    lm_sample_frac=lm_sample_frac,
                    random_state=random_state,
                )

        return precomp_vals

    @classmethod
    def precompute_landmarking_kfolds(
        cls,
        N: np.ndarray,
        y: t.Optional[np.ndarray] = None,
        num_cv_folds: int = 10,
        shuffle_cv_folds: t.Optional[bool] = False,
        random_state: t.Optional[int] = None,
        lm_sample_frac: float = 1.0,
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute k-fold cross validation related values.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`, optional
            Target attribute.

        num_cv_folds : int, optional
            Number of folds to k-fold cross validation.

        shuffle_cv_folds : bool, optional
            If True, shuffle the samples before splitting the k-fold cross
            validation.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        lm_sample_frac : float, optional
            The percentage of examples subsampled. Value different from default
            will generate the subsampling-based relative landmarking
            metafeatures. Used only if ``sample_inds`` is not precomputed and
            if ``shuffle_cv_folds`` is False or ``random_state`` is given.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``skf`` (:obj:`sklearn.model_selection.StratifiedKFold`):
                  Stratified K-Folds cross-validator. Provides train/test
                  indices to split data in train/test sets.
        """
        precomp_vals = {}

        if N is not None and N.size > 0 and y is not None:
            if "skf" not in kwargs:
                precomp_vals["skf"] = sklearn.model_selection.KFold(
                    n_splits=num_cv_folds,
                    random_state=random_state if shuffle_cv_folds else None,
                )

            if (
                not shuffle_cv_folds
                or random_state is not None
                and "cv_folds_imp_rank" not in kwargs
            ):
                skf = precomp_vals.get("skf", kwargs.get("skf"))
                sample_inds = kwargs.get("sample_inds")

                N, y = cls._sample_data(
                    N=N,
                    y=y,
                    lm_sample_frac=lm_sample_frac,
                    random_state=random_state,
                    sample_inds=sample_inds,
                )

                attr_fold_imp = np.array(
                    [
                        cls._rank_feat_importance(
                            N=N[inds_train, :],
                            y=y[inds_train],
                            random_state=random_state,
                        )
                        for inds_train, inds_test in skf.split(N, y)
                    ],
                    dtype=int,
                )

                precomp_vals["cv_folds_imp_rank"] = attr_fold_imp

        return precomp_vals

    @classmethod
    def _get_sample_inds(
        cls,
        num_inst: int,
        lm_sample_frac: float,
        random_state: t.Optional[int],
    ) -> np.ndarray:
        """Sample indices to calculate subsampling landmarking metafeatures."""
        if random_state is not None:
            np.random.seed(random_state)

        sample_inds = np.random.choice(
            a=num_inst, size=int(lm_sample_frac * num_inst), replace=False
        )

        return sample_inds

    @classmethod
    def _sample_data(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        lm_sample_frac: float,
        random_state: t.Optional[int] = None,
        sample_inds: t.Optional[np.ndarray] = None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Select ``lm_sample_frac`` percent of data from ``N`` and ``y``."""
        if lm_sample_frac >= 1.0 and sample_inds is None:
            return N, y

        if sample_inds is None:
            num_inst = y.size

            sample_inds = cls._get_sample_inds(
                num_inst=num_inst,
                lm_sample_frac=lm_sample_frac,
                random_state=random_state,
            )

        return N[sample_inds, :], y[sample_inds]

    @classmethod
    def _rank_feat_importance(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        lm_sample_frac: float = 1.0,
        sample_inds: t.Optional[np.ndarray] = None,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Rank the feature importances of a DT model.

        It is used the ``sklearn.tree.DecisionTreeClassifier``
        implementation.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        lm_sample_frac : float, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        Returns
        -------
        :obj:`np.ndarray`
            Ranking of the decision tree features importance.
        """
        N, y = cls._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds,
        )

        clf = DecisionTreeRegressor(
            random_state=random_state
        ).fit(N, y)

        return np.argsort(clf.feature_importances_)

    @classmethod
    def get_best_node_regr(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray] = r2_score,
        skf: t.Optional[sklearn.model_selection.StratifiedKFold] = None,
        num_cv_folds: int = 10,
        lm_sample_frac: float = 1.0,
        sample_inds: t.Optional[np.ndarray] = None,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Performance of a the best single decision tree node.

        Construct a single decision tree node model induced by the most
        informative attribute to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`sklearn.model_selection.StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        num_cv_folds : int, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        shuffle_cv_folds : bool, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : float, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        Returns
        -------
        :obj:`np.ndarray`
            The Decision Tree best-node model performance of each fold.

        References
        ----------
        .. [1] Hilan Bensusan and Christophe Giraud-Carrier. Discovering task
           neighbourhoods through landmark learning performances. In 4th
           European Conference on Principles of Data Mining and Knowledge
           Discovery (PKDD), pages 325 – 330, 2000.
        .. [2] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = cls._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds,
        )

        if skf is None:
            skf = sklearn.model_selection.KFold(
                n_splits=num_cv_folds,
                random_state=random_state
            )

        model = DecisionTreeRegressor(
            max_depth=1, random_state=random_state
        )

        res = np.zeros(skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = score(y_test, y_pred)

        return res

    @classmethod
    def get_random_node_regr(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray] = r2_score,
        skf: t.Optional[sklearn.model_selection.StratifiedKFold] = None,
        num_cv_folds: int = 10,
        lm_sample_frac: float = 1.0,
        sample_inds: t.Optional[np.ndarray] = None,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Performance of the single decision tree node model induced by a
        random attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`sklearn.model_selection.StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        num_cv_folds : int, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : float, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        Returns
        -------
        :obj:`np.ndarray`
            The Decision Tree random-node model performance of each fold.

        References
        ----------
        .. [1] Hilan Bensusan and Christophe Giraud-Carrier. Discovering task
           neighbourhoods through landmark learning performances. In 4th
           European Conference on Principles of Data Mining and Knowledge
           Discovery (PKDD), pages 325 – 330, 2000.
        .. [2] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = cls._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds,
        )

        if skf is None:
            skf = sklearn.model_selection.KFold(
                n_splits=num_cv_folds,
                random_state=random_state
            )

        if random_state is not None:
            np.random.seed(random_state)

        rand_ind_attr = np.random.randint(0, N.shape[1], size=1)

        model = DecisionTreeRegressor(
            max_depth=1, random_state=random_state
        )

        res = np.zeros(skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(skf.split(N, y)):
            X_train = N[inds_train, rand_ind_attr, np.newaxis]
            X_test = N[inds_test, rand_ind_attr, np.newaxis]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = score(y_test, y_pred)
        print(res)
        return res

    @classmethod
    def get_worst_node_regr(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray] = r2_score,
        skf: t.Optional[sklearn.model_selection.StratifiedKFold] = None,
        num_cv_folds: int = 10,
        lm_sample_frac: float = 1.0,
        sample_inds: t.Optional[np.ndarray] = None,
        random_state: t.Optional[int] = None,
        cv_folds_imp_rank: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Performance of the single decision tree node model induced by the
        worst informative attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`sklearn.model_selection.StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        num_cv_folds : int, optional
            Number of folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : bool, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : float, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        cv_folds_imp_rank : :obj:`np.ndarray`, optional
            Ranking based on the predictive attribute importance per
            cross-validation fold. The rows correspond to each fold,
            and the columns correspond to each predictive attribute.
            Argument used to take advantage of precomputations. Do not
            use it if the k-fold cross validation splitter shuffles the
            data with no random seed fixed.

        Returns
        -------
        :obj:`np.ndarray`
            The Decision Tree worst-node model performance of each fold.

        References
        ----------
        .. [1] Hilan Bensusan and Christophe Giraud-Carrier. Discovering task
           neighbourhoods through landmark learning performances. In 4th
           European Conference on Principles of Data Mining and Knowledge
           Discovery (PKDD), pages 325 – 330, 2000.
        .. [2] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = cls._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds,
        )

        if skf is None:
            skf = sklearn.model_selection.KFold(
                n_splits=num_cv_folds,
                random_state=random_state
            )

        model = DecisionTreeRegressor(
            max_depth=1, random_state=random_state
        )

        res = np.zeros(skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(skf.split(N, y)):
            if cv_folds_imp_rank is not None:
                imp_rank = cv_folds_imp_rank[ind_fold, :]
            else:

                imp_rank = cls._rank_feat_importance(
                    N=N[inds_train, :],
                    y=y[inds_train],
                    random_state=random_state,
                )

            X_train = N[inds_train, imp_rank[0], np.newaxis]
            X_test = N[inds_test, imp_rank[0], np.newaxis]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = score(y_test, y_pred)

        return res

    @classmethod
    def get_linear_regr(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray] = r2_score,
        skf: t.Optional[sklearn.model_selection.StratifiedKFold] = None,
        num_cv_folds: int = 10,
        lm_sample_frac: float = 1.0,
        sample_inds: t.Optional[np.ndarray] = None,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Performance of the Linear Discriminant classifier.

        The Linear Discriminant Classifier is used to construct a linear split
        (non parallel axis) in the data to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`sklearn.model_selection.StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        num_cv_folds : int, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : bool, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : float, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        Returns
        -------
        :obj:`np.ndarray`
            The Linear Discriminant Analysis model performance of each fold.

        References
        ----------
        .. [1] Hilan Bensusan and Christophe Giraud-Carrier. Discovering task
           neighbourhoods through landmark learning performances. In 4th
           European Conference on Principles of Data Mining and Knowledge
           Discovery (PKDD), pages 325 – 330, 2000.
        .. [2] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = cls._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds,
        )

        if skf is None:
            skf = sklearn.model_selection.KFold(
                n_splits=num_cv_folds,
                random_state=random_state
            )

        model = LinearRegression()

        res = np.zeros(skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = score(y_test, y_pred)

        return res

    @classmethod
    def get_bayesian_ridge(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray] = r2_score,
        skf: t.Optional[sklearn.model_selection.StratifiedKFold] = None,
        num_cv_folds: int = 10,
        lm_sample_frac: float = 1.0,
        sample_inds: t.Optional[np.ndarray] = None,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Performance of the Naive Bayes classifier.

        It assumes that the attributes are independent and each example
        belongs to a certain class based on the Bayes probability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`sklearn.model_selection.StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        num_cv_folds : int, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : bool, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : float, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : int, optional
            If given, set the random seed before any pseudo-random calculations
            to keep the experiments reproducible.

        Returns
        -------
        :obj:`np.ndarray`
            The Naive Bayes model performance of each fold.

        References
        ----------
        .. [1] Hilan Bensusan and Christophe Giraud-Carrier. Discovering task
           neighbourhoods through landmark learning performances. In 4th
           European Conference on Principles of Data Mining and Knowledge
           Discovery (PKDD), pages 325 – 330, 2000.
        .. [2] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = cls._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds,
        )

        if skf is None:
            skf = sklearn.model_selection.KFold(
                n_splits=num_cv_folds,
                random_state=random_state
            )

        model = BayesianRidge()

        res = np.zeros(skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = score(y_test, y_pred)

        return res

    