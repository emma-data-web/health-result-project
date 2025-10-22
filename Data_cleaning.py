import joblib
import pandas as pd
import seaborn as sns


data = pd.read_csv("estimated_numbers.csv")

data["Country"].value_counts().count()

sns.heatmap(data.isnull())


data["death"] = data["No. of deaths"].apply(lambda x: x.split("[")[0])

data["case"] = data["No. of cases"].apply(lambda x: x.split("[")[0])

#the data death and case columns are messy, you can use a different apporach though

joblib.dump(data,"estimated_numbers.csv")