import numpy as np
import os
import pandas as pd


def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, dataset + ".csv"))
    total_count = df.shape[0]
    positive_counts = [df['Nodule'].sum(), df['Pneumothorax'].sum()]
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts                                                
