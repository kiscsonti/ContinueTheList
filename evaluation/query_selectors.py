import numpy as np


def select_min(querries):
    return querries[0]


def select_median(querries, threshold):

    thresholded_querries = list(filter(lambda x: x[1] <= threshold, querries))
    index = int(len(thresholded_querries)/2)
    return querries[index]


def select_average(querries, threshold):

    thresholded_querries = list(filter(lambda x: x[1] <= threshold, querries))
    avg = np.mean([item[1] for item in thresholded_querries])
    closest_item = None
    closest_diff = None

    # CLOSEST TO AVG
    for item in thresholded_querries:
        if closest_item is None:
            closest_item = item
            closest_diff = abs(avg-item[1])
            continue
        if closest_diff > abs(avg-item[1]):
            closest_item = item
            closest_diff = abs(avg-item[1])

    return closest_item


def select_thresholded(querries, threshold):
    prev_query = None
    for item in querries:
        if item[1] >= threshold:
            if prev_query is None:
                return item
            else:
                return prev_query

        prev_query = item
    return None


def select_xth(querries, x):
    rank = 0
    size = None
    for item in querries:
        if size is None or item[1] != size:
            rank += 1
            size = item[1]

        if rank == x:
            return item
    return None
