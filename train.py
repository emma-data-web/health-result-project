import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from modules import ColumnLabelEncoder

df = pd.read_csv("ai_health_dataset.csv")


X = df.drop("disease", axis=1)
y = df["disease"]

label_encoder = ColumnLabelEncoder()
y_encoded = label_encoder.fit_transform(y)


numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    ))
])


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)


score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.3f}")


joblib.dump(pipeline, "health_disease_pipeline.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")

print(" Model and encoder saved successfully!")
