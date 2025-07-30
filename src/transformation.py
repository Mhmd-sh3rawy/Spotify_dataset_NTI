import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def key_cyclic_encoding(df, column) -> None:
    """Encoding the 'key' column and add two columns in dataframe."""
    df[str(column)+'_sin'] = np.sin(2 * np.pi * df[column]/12)
    df[str(column)+"_cos"] = np.cos(2 * np.pi * df[column]/12)
    print(df[[column, str(column)+'_sin', str(column)+"_cos"]].sample(3), "\n")

