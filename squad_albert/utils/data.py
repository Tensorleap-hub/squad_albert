import pandas as pd


def most_common_titles(ds, k: int = 5):
    return list(pd.Series(ds.data['title']).value_counts()[:k].keys())
