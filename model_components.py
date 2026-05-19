import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


class ConsensusFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, chi2_alpha=0.05, min_votes=2, min_selected=3):
        self.chi2_alpha = chi2_alpha
        self.min_votes = min_votes
        self.min_selected = min_selected

    def fit(self, X, y):
        n_features = X.shape[1]
        rfe_k = max(1, int(math.sqrt(n_features)))

        X_mm = MinMaxScaler().fit_transform(X)
        _, chi2_pvals = chi2(X_mm, y)
        mask_chi2 = chi2_pvals < self.chi2_alpha

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight='balanced'
        )
        rf.fit(X, y)
        mask_rf = rf.feature_importances_ > rf.feature_importances_.mean()

        X_rfe = RobustScaler().fit_transform(X)
        lr = LogisticRegression(
            max_iter=300,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            C=0.1
        )
        step = max(1, n_features // 10)
        rfe = RFE(estimator=lr, n_features_to_select=rfe_k, step=step)
        rfe.fit(X_rfe, y)
        mask_rfe = rfe.support_

        votes = mask_chi2.astype(int) + mask_rf.astype(int) + mask_rfe.astype(int)
        support = votes >= self.min_votes

        if support.sum() < self.min_selected:
            support = votes >= 1
        if support.sum() == 0:
            top_idx = np.argsort(votes)[::-1][:max(3, n_features // 10)]
            support = np.zeros(n_features, dtype=bool)
            support[top_idx] = True

        self.support_ = support
        self.votes_ = votes
        self.n_selected_ = int(support.sum())
        return self

    def transform(self, X):
        return X[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_


class P3SelectorConsensus(BaseEstimator, TransformerMixin):
    def __init__(self, n_p1, min_votes=2):
        self.n_p1 = n_p1
        self.min_votes = min_votes

    def fit(self, X, y):
        self.fs_ = ConsensusFeatureSelector(min_votes=self.min_votes)
        self.fs_.fit(X[:, self.n_p1:], y)
        return self

    def transform(self, X):
        return np.hstack([X[:, :self.n_p1], self.fs_.transform(X[:, self.n_p1:])])

    def get_support(self):
        return np.concatenate([np.ones(self.n_p1, dtype=bool), self.fs_.get_support()])
