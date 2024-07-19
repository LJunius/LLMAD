# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import heapq
import pandas as pd
from scipy.stats import scoreatpercentile

import pandas as pd
import json
from matplotlib import pyplot as plt
from matplotlib import gridspec
from PIL import Image
import os
from tqdm import tqdm

def affine_transform(time_series, alpha, beta):
    # Calculate the 1% and 99% percentiles
    min_val = np.percentile(time_series, 0.1)
    max_val = np.percentile(time_series, 99)
    b = min_val - beta * (max_val - min_val)
    
    # Shift the series
    shifted_series = time_series - b
    
    # Calculate the value of 'a' using scipy's scoreatpercentile function to find the alpha percentile
    a = scoreatpercentile(shifted_series, alpha)
    
    # If the values in the series are very small, limit 'a' to a maximum of 0.01
    if np.all(np.abs(shifted_series) < 1e-3):
        a = min(a, 0.01)
    
    # Apply the affine transformation
    transformed_series = shifted_series / a
    
    # If the minimum value in the original series is 0, keep the 0 position unchanged after transformation
    transformed_series[time_series == 0] = 0
    
    return transformed_series


def fast_dtw_distance(series1, series2, dist_div_len=False):
    series1 = series1.astype(float)
    series2 = series2.astype(float)

    distance, path = fastdtw(series1, series2, dist=lambda x, y: np.linalg.norm(x - y))
    if dist_div_len:
        avg_length = (len(series1) + len(series2)) / 2
        distance = distance / avg_length

    return distance

import heapq


# Define a function to find the most similar series to a given sequence
def find_most_similar_series_fast(X, T_list, top_k=1, dist_div_len=False):
    # Use a heap to store the smallest k scores and the corresponding series
    heap = []
    
    # Iterate over all sequences
    for idx, Y in enumerate(T_list):
        score = fast_dtw_distance(X, Y, dist_div_len)  # Assume this function is already defined
        
        # If the heap size is less than top_k, directly add
        if len(heap) < top_k:
            heapq.heappush(heap, (-score, idx, Y))
        else:
            # If the current score is smaller than the largest score in the heap, replace it
            heapq.heappushpop(heap, (-score, idx, Y))
    
    # Extract the top_k series, scores, and indices
    top_k_series = []
    top_k_scores = []
    top_k_indices = []
    
    # Sort the elements in the heap in ascending order of scores
    for score, idx, series in sorted(heap, reverse=True):
        top_k_scores.append(-score)  # Convert the score back to a positive value
        top_k_indices.append(idx)
        top_k_series.append(series)
    
    return top_k_series, top_k_scores, top_k_indices


def find_zero_sequences(data, min_len=100, max_len=800, overlap=0):
    zero_sequences = []
    start_index = None

    for index, row in data.iterrows():
        if row['label'] == 0:
            if start_index is None:
                start_index = index
        else:
            if start_index is not None and index - start_index >= min_len:
                zero_sequences.append(data.iloc[start_index:index][['value']])
                start_index = None
            else:
                start_index = None

        if start_index is not None and index - start_index + 1 >= max_len:
            end_index = index
            if end_index - start_index + 1 >= min_len:
                zero_sequences.append(data.iloc[start_index:end_index + 1][['value']])
            start_index = index - overlap + 1 if overlap > 0 else index + 1

    if start_index is not None and len(data) - start_index >= min_len:
        zero_sequences.append(data.iloc[start_index:][['value']])

    return zero_sequences


def find_anomalies(data, pad_len=5, max_len=800):
    non_zero_sequences = []  # Used to store the values of anomaly segments
    sequence_strings = []  # Used to store the string representations of anomaly segments
    label_sequences = []  # Used to store the labels of anomaly segments

    current_values = []  # Values of the current anomaly segment
    current_labels = []  # Labels of the current anomaly segment
    current_string = ""  # String representation of the current anomaly segment
    in_anomaly = False  # Flag indicating whether an anomaly segment is being recorded

    for index, row in data.iterrows():
        # For each anomaly point
        if row['label'] == 1:
            if not in_anomaly:
                # Start recording a new anomaly segment, add left padding
                in_anomaly = True
                start_pad_index = max(0, index - pad_len)
                try:
                    for i in range(start_pad_index, index):
                        value = data.at[i, 'value']
                        label = data.at[i, 'label']
                        current_values.append(value)
                        current_labels.append(label)
                        current_string += "{},".format("*{}*".format(int(value)) if label == 1 else value)
                except:
                    import pdb; pdb.set_trace()
            # Add to the current anomaly segment
            current_values.append(row['value'])
            current_labels.append(row['label'])
            current_string += "*{}*,".format(int(row['value']))

            # Check if the maximum length limit is reached
            if len(current_values) >= max_len:
                # Trim the string to remove the last comma and save the current anomaly segment
                current_string = current_string.rstrip(',')
                # If current_labels are not all 1
                if 0 in current_labels:    
                    non_zero_sequences.append(current_values)
                    sequence_strings.append(current_string)
                    label_sequences.append(current_labels)

                # Reset the current anomaly segment, prepare to record a new anomaly segment
                current_values = []
                current_labels = []
                current_string = ""
                in_anomaly = False

        else:
            # If the current anomaly segment is not empty and there was a previous anomaly point
            if in_anomaly:
                # Add right padding
                end_pad_index = min(len(data), index + pad_len)
                for i in range(index, end_pad_index):
                    value = data.at[i, 'value']
                    label = data.at[i, 'label']
                    current_values.append(value)
                    current_labels.append(label)
                    current_string += "{},".format("*{}*".format(int(value)) if label == 1 else int(value))

                # Trim the string to remove the last comma and save the current anomaly segment
                current_string = current_string.rstrip(',')
                if 0 in current_labels:   
                    non_zero_sequences.append(current_values)
                    sequence_strings.append(current_string)
                    label_sequences.append(current_labels)

                # Reset the current anomaly segment
                current_values = []
                current_labels = []
                current_string = ""
                in_anomaly = False

    # Check if there are any unsaved anomaly segments
    if in_anomaly:
        # Add right padding
        end_pad_index = min(len(data), index + 1 + pad_len)
        for i in range(index + 1, end_pad_index):
            value = data.at[i, 'value']
            label = data.at[i, 'label']
            current_values.append(value)
            current_labels.append(label)
            current_string += "{},".format("*{}*".format(int(value)) if label == 1 else int(value))

        # Trim the string to remove the last comma and save the current anomaly segment
        current_string = current_string.rstrip(',')
        if 0 in current_labels:   
            non_zero_sequences.append(current_values)
            sequence_strings.append(current_string)
            label_sequences.append(current_labels)

    return non_zero_sequences, sequence_strings, label_sequences


    

if __name__ == "__main__":
    X = [1, 2, 3, 4, 5]

    T_list = [
        [1, 2, 3, 4],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5], 
        [10, 9, 8, 7, 6]
    ]
    X = np.squeeze(np.asarray(X))

    T_list = [np.squeeze(np.asarray(series)) for series in T_list]

    top_k_series, top_k_scores, top_k_indices = find_most_similar_series_fast(X, T_list, top_k=2)

    print("Most similar series:", top_k_series)
    print("DTW distance:", top_k_scores)
    print("Indices:", top_k_indices)
    
