"""Classical machine learning models.

Traditional ML models for biomarker classification and regression:
- Random Forest
- Support Vector Machines (SVM)
- Logistic Regression
- XGBoost
- Gradient Boosting Machines
"""

from omicselector2.models.classical.linear_models import (
    LogisticRegressionModel,
    SVMClassifier,
)
from omicselector2.models.classical.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from omicselector2.models.classical.xgboost_models import (
    XGBoostClassifier,
    XGBoostRegressor,
)

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "LogisticRegressionModel",
    "SVMClassifier",
    "XGBoostClassifier",
    "XGBoostRegressor",
]
