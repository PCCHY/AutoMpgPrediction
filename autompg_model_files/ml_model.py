import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor

def preprocess_origin_cols(df):
    df['origin'] = df['origin'].map({1:'India',2:'USA',3:'Germany'})
    return df

acc_ix, hpower_ix, cyl_ix = 4,2,1

class CustomAttrAdder(BaseEstimator, TransformerMixin):

    def __init__(self, acc_on_power = True):
        self.acc_on_power = acc_on_power

    def fit(self, X, y=None):
        return self

    def transform(self,X):
        acc_on_cyl = X[:,acc_ix]/X[:,cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:,acc_ix] / X[:,hpower_ix]
            return np.c_[X,acc_on_power,acc_on_cyl]
        return np.c_[X,acc_on_cyl]
    
def num_pipeline_transformer(df):
    numerics = ['float64','int64']
    num_attrs = df.select_dtypes(include = numerics)

    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler',StandardScaler())
        ])

    return num_attrs, num_pipeline

def pipeline_transformer(df):
    cat_attrs = ['origin']
    num_attrs, num_pipeline = num_pipeline_transformer(df)

    full_pipeline = ColumnTransformer([
        ('num',num_pipeline, list(num_attrs)),
        ('cat',OneHotEncoder(), cat_attrs)
        ])

    prepared_df = full_pipeline.fit_transform(df)
    return prepared_df

def predict_mpg(config, model):
    input_df = pd.DataFrame.from_dict(config)

    preproc_df = preprocess_origin_cols(input_df)
    prepared_df = pipeline_transformer(preproc_df)

    y_pred = model.predict(prepared_df)
    return y_pred
