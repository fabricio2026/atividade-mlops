MODEL_PATH = "models/bank_marketing_model.joblib"
API_VERSION = "1.0.0"
MODEL_ALGORITHM = "RandomForestClassifier"

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome",
]

NUMERICAL_FEATURES = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx",
    "euribor3m", "nr_employed",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
