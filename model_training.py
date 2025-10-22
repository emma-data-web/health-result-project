import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
import joblib


df = pd.read_csv("estimated_numbers.csv")

columns = ['Country', 'Year',
       'No. of cases_median', 'No. of cases_min', 'No. of cases_max',
       'No. of deaths_median', 'No. of deaths_min', 'No. of deaths_max',
       'WHO Region', 'case']

target = "death"

numerical_col = ["Year","No. of cases_median","No. of cases_min",
                 "No. of cases_max","No. of deaths_median",
                 "No. of deaths_min","No. of deaths_max","case"]

categorical_col = ["Country","WHO Region"]


numerical_pipeline = Pipeline(steps=[
    ("num_process",SimpleImputer(strategy="mean"))
])

categorical_pipeline = Pipeline(steps=[
    ("cat_process",OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
]) 


final_cols = ColumnTransformer(transformers=[
    ("numerical",numerical_pipeline,numerical_col),
    ("categorical",categorical_pipeline,categorical_col)
], verbose=2, remainder="passthrough", force_int_remainder_cols=False)


model = LGBMRegressor()

final_pipeline = Pipeline(steps=[
    ("preprocessing",final_cols),
    ("model", model)
])

x_train, x_test, y_train, y_test = train_test_split(df[columns],df[target],test_size=0.3, random_state=101)

final_pipeline.fit(x_train,y_train)

joblib.dump(final_pipeline,"malaria_death_mmodel.pkl")