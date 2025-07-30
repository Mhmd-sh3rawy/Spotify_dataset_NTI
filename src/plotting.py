import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def distribution_reform(df:pd.DataFrame, columns: list[str], log_apply=None, sqrt_apply=None) -> None:
    """
    Plots the distribution and transformation of skewed features in a dataset.
    """
    for col in columns:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        fig.suptitle(f'Distribution of "{col}" and Transformations', fontsize=14, fontweight='bold')

        # Original distribution
        sns.histplot(df[col], ax=axes[0], kde=True, color='orange', edgecolor='black')
        skew_value = df[col].skew()
        axes[0].set_title(f'Original Distribution\nSkew: {skew_value:.4f}')
        axes[0].set_xlabel(col)

        # Transformed distribution
        if log_apply == True:
            transformed = np.log1p(df[col] + 0.0000000001)
            transform_label = 'Log1p'
        elif sqrt_apply == True:
            transformed = np.sqrt(df[col])
            transform_label = 'Square Root'
        else:
            transformed = df[col]
            transform_label = 'No Transformation'

        transformed_skew = transformed.skew()
        sns.histplot(transformed, ax=axes[1], kde=True, color='skyblue', edgecolor='black')
        axes[1].set_title(f'{transform_label} Transformation\nSkew: {transformed_skew:.4f}')
        axes[1].set_xlabel(f"{col} ({transform_label})")

        plt.tight_layout()
        plt.show()