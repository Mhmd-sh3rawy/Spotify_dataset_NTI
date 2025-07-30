import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def describe_with_percentiles_show_kdeplot(df,column):
    """show percentile with kde plot."""
    print(f"skewness: {df[column].skew()}")
    print('--'*20)
    print(df[column].describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]))
    print('--'*20)
    sns.kdeplot(df[column])