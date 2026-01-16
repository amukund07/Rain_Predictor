# Importing modules 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder 
from sklearn.linear_model import LogisticRegression 

# Import dataset
df = pd.read_csv("weatherAUS.csv")

# Drop useless columns
df = df.drop(["Evaporation", "Sunshine","WindDir9am","WindDir3pm","WindSpeed3pm","Cloud9am","Cloud3pm","WindSpeed9am","Temp9am","Temp3pm"], axis=1)


# Drop rows with missing target values
df.dropna(subset=['RainToday', 'RainTomorrow',], inplace=True)

year= pd.to_datetime(df.Date).dt.year
train_df= df[year<2015]
val_df= df[year==2015]
test_df= df[year>2015]

# Setting up input and output targest
input_cols = list((train_df.columns)[1:-1])
target_col = 'RainTomorrow'


# Make a copy 
train_inputs= train_df[input_cols].copy()
train_target= train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# To idenitfy numric and catagorical data
numeric_cols= train_inputs.select_dtypes(include=np.number).columns.tolist()
cato_cols= train_inputs.select_dtypes("object").columns.tolist()


# Imputning missing data
imputer= SimpleImputer(strategy="mean")
imputer.fit(df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Standarization of data
scaler= MinMaxScaler()
scaler.fit(df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# One hot Encoding
encoder= OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(df[cato_cols])
encoded_cols = list(encoder.get_feature_names_out(cato_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[cato_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[cato_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[cato_cols])

# Combining the DF
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# Traning model


model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    class_weight={"No": 1, "Yes": 1.5}
)

model.fit(X_train, train_target)


cat_imputer = SimpleImputer(strategy="most_frequent")
cat_imputer.fit(train_inputs[cato_cols])

train_inputs[cato_cols] = cat_imputer.transform(train_inputs[cato_cols])
val_inputs[cato_cols] = cat_imputer.transform(val_inputs[cato_cols])
test_inputs[cato_cols] = cat_imputer.transform(test_inputs[cato_cols])


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[cato_cols] = cat_imputer.transform(input_df[cato_cols])
    input_df[encoded_cols] = encoder.transform(input_df[cato_cols])

    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob



import pickle

with open("rain_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("cat_imputer.pkl", "wb") as f:
    pickle.dump(cat_imputer, f)

with open("columns.pkl", "wb") as f:
    pickle.dump({
        "numeric_cols": numeric_cols,
        "cato_cols": cato_cols,
        "encoded_cols": encoded_cols
    }, f)
