import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LogisticRegression
import joblib
import gzip

#changing dtypes to consume less memory
df = pd.read_csv('data/heart.csv', dtype={'HeartDisease': 'category', 'Smoking': 'category', 'AlcoholDrinking': 'category',
                                      'Stroke': 'category', 'DiffWalking': 'category', 'Sex': 'category',
                                      'AgeCategory': 'category', 'Race': 'category', 'Diabetic': 'category',
                                      'PhysicalActivity': 'category', 'GenHealth': 'category', 'Asthma': 'category',
                                      'KidneyDisease': 'category',  'SkinCancer': 'category', 
                                      "BMI": "float32", 'PhysicalHealth': 'int8', 
                                      'MentalHealth': 'int8', 'SleepTime': 'int8',
                                      })
df.head()

df.drop(['Race'], axis=1, inplace=True)

# Separate target from predictors
y_before = df.HeartDisease
X = df.drop(['HeartDisease'], axis=1)

#transform y_before into 0s and 1s
y_after = pd.get_dummies(y_before)
y_after = y_after.drop(["No"], axis=1) #drop column No, so 0->No and 1->Yes
y = y_after.rename(columns={"Yes": "HeartDisease"}) #renaming Yes column to original column name

# Divide data into training and validation subsets
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=42, stratify=y)

# Selecting categorical columns
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].dtype.name in ["category"]] 

# Selecting numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int8', 'float32']]


# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy() #test data

#SMOTENC oversampling to fix the imbalance
smote_nc = SMOTENC(categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], sampling_strategy=0.5,
                   random_state=42, n_jobs=-1)
X_train, y_train= smote_nc.fit_resample(X_train, y_train)

# Preprocessing data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[('scaler', RobustScaler())])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ])

#pipeline with best hyperparameters
lr = Pipeline(steps=[('preprocessor', preprocessor),
                     ('model',LogisticRegression(solver= "newton-cg", C= 1000, random_state=42)),
                             ])
lr_tuned = lr.fit(X_train, y_train.values.ravel())

# Export model
joblib.dump(lr_tuned, gzip.open('model_to_run/model.dat.gz', "wb"))