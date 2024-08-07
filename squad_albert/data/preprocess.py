import pandas as pd
from enum import Enum
from typing import Dict, Tuple
import numpy as np
from datasets import load_dataset
from squad_albert.utils.data import most_common_titles
from squad_albert.config import persistent_dir, CONFIG


def load_data() -> Tuple[np.ndarray, dict, np.ndarray, dict, Dict[str, Enum]]:
    """
    Description: Loads the SQuAD dataset and splits it into training and validation sets based on predefined titles.
    Returns:
    train_idx (np.ndarray): Indices of samples in the training set.
    train_ds (dict): Training dataset containing the samples.
    val_idx (np.ndarray): Indices of samples in the validation set.
    val_ds (dict): Validation dataset containing the samples.
    enums_dic (Dict[str, Enum]): A dictionary with an enumeration for the "title" field containing possible values from training and validation titles.
    """
    np.random.seed(0)
    K = CONFIG["k_titles"]
    #ds = load_dataset('squad') #if run locally
    ds = load_dataset('squad', cache_dir=persistent_dir)
    train_ds, val_ds = ds['train'], ds['validation']

    train_idx, val_idx, test_idx = [], [], []
    train_titles, val_titles = most_common_titles(train_ds, K), most_common_titles(val_ds, K*2)

    for col in train_titles:
        idx = list(np.where(pd.Series(train_ds.data['title']) == col)[0])
        train_idx += idx
    for col in val_titles[:K]:
        idx = list(np.where(pd.Series(val_ds.data['title']) == col)[0])
        val_idx += idx
    for col in val_titles[K:]:
        idx = list(np.where(pd.Series(val_ds.data['title']) == col)[0])
        test_idx += idx

    train_idx = np.random.choice(train_idx, size=CONFIG['TRAIN_SIZE'], replace=False) if CONFIG['TRAIN_SIZE'] < len(train_idx) else train_idx
    val_idx = np.random.choice(val_idx, size=CONFIG['VAL_SIZE'], replace=False) if CONFIG['VAL_SIZE'] < len(val_idx) else val_idx
    test_idx = np.random.choice(test_idx, size=CONFIG['TEST_SIZE'], replace=False) if CONFIG['TEST_SIZE'] < len(test_idx) else test_idx

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    enums_dic = {}
    for col in ["title"]:  # , "context"]:
        enum = Enum(col, train_titles + val_titles)
        enums_dic[col] = enum

    return train_idx, train_ds, val_idx, val_ds, test_idx, val_ds, enums_dic