import numpy as np
from collections import Counter
import math

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    """
    target_column = data[:, -1]  # last column = target
    values, counts = np.unique(target_column, return_counts=True)
    probabilities = counts / counts.sum()

    entropy = 0
    for p in probabilities:
        if p > 0:  # avoid log2(0)
            entropy -= p * math.log2(p)
    return entropy


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    """
    total_rows = len(data)
    attribute_values, counts = np.unique(data[:, attribute], return_counts=True)

    avg_info = 0
    for value, count in zip(attribute_values, counts):
        subset = data[data[:, attribute] == value]  # filter rows
        subset_entropy = get_entropy_of_dataset(subset)
        weight = count / total_rows
        avg_info += weight * subset_entropy
    return avg_info


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    """
    total_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    info_gain = total_entropy - avg_info
    return round(info_gain, 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    """
    n_attributes = data.shape[1] - 1  # exclude target column
    gains = {}
    for attr in range(n_attributes):
        gains[attr] = get_information_gain(data, attr)

    best_attr = max(gains, key=gains.get)
    return gains, best_attr