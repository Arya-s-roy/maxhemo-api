import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HbFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # ----------------------------------------------------
        # Gender encoding
        # Male = 1
        # Female = 0
        # ----------------------------------------------------
        X["Gender"] = X["Gender"].map({
            "Male": 1,
            "Female": 0
        })

        # ----------------------------------------------------
        # Engineered features
        # ----------------------------------------------------

        X["IR_Red_Ratio"] = (
            X["Infra Red (a.u)"] /
            X["Red (a.u)"]
        )

        X["Red_IR_Ratio"] = (
            X["Red (a.u)"] /
            X["Infra Red (a.u)"]
        )

        X["IR_Red_Diff"] = (
            X["Infra Red (a.u)"] -
            X["Red (a.u)"]
        )

        X["IR_Red_Sum"] = (
            X["Infra Red (a.u)"] +
            X["Red (a.u)"]
        )

        X["Normalized_Difference"] = (
            (
                X["Infra Red (a.u)"] -
                X["Red (a.u)"]
            )
            /
            (
                X["Infra Red (a.u)"] +
                X["Red (a.u)"]
            )
        )

        X["Log_IR_Red_Ratio"] = np.log(
            X["Infra Red (a.u)"] /
            X["Red (a.u)"]
        )

        return X