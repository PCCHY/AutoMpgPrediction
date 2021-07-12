import pandas as pd
import numpy as np
from unittest.mock import patch
from autompg_model_files.ml_model import pipeline_transformer 
import pytest

def test_pipeline():
    data = pd.DataFrame(
        [[18,8,307,130,3504,12,70,1,"chevrolet chevelle malibu"],
         [15,8,350,165,3693,11.5,70,1,"buick skylark 320"],
         [18,8,318,np.nan,3436,11,70,1,"plymouth satellite"],
         [16,8,304,150,3433,12,70,1,"amc rebel sst"],
         [17,8,302,140,3449,10.5,70,1,"ford torino"]],
        columns = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model","origin","car"]
        )
    df = pipeline_transformer(data)
    print(df[2])
    assert isinstance(df,np.ndarray)
    assert not np.any(np.isnan(df))
