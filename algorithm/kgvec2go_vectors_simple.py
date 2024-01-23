from gensim.models import Word2Vec, KeyedVectors
from gensim import models
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import RadiusNeighborsClassifier
from copy import copy
from excludes.filters import filter_none, filter_url


def load_gensim_model(path):
    w2v_model = models.KeyedVectors.load(
        path, mmap="r"
    )
    return w2v_model


def ellipse_rectangle_dist(center, radius, point, eps=1e-4):
    res = np.max(np.power(np.divide(np.subtract(center, point), radius), 2))
    # res = np.sum(np.power(np.divide(np.subtract(center, point), radius), 2))
    return res


def get_radius(center, seed):
    return np.max(np.abs(np.array(seed) - np.array(center)), axis=0)


def run_method(seed, wv_model: KeyedVectors, distance="circle"):
    """
    Input is a single record's seed.
    distance can be the following: ["circle", "euclidean", "ellipse", "l1ellipse"]
    :return:
    """

    vectors = dict()
    missing = 0
    for item in seed:
        try:
            vectors[item] = list(wv_model[item])
        except Exception as e:
            missing += 1
    if missing == len(seed):
        return None, missing
    center_vector = np.mean(list(vectors.values()), axis=0)
    closer_elements = list()

    if distance == "circle" or distance == "euclidean":
        point_dist = cdist([center_vector], list(vectors.values()), metric="euclidean")
        threshold = np.max(point_dist)
        all_euclidean_dist = cdist([center_vector], wv_model.vectors, metric="euclidean")
        for index in all_euclidean_dist.argsort()[0]:
            if all_euclidean_dist[0][index] > threshold:
                break
            closer_elements.append((index, all_euclidean_dist[0][index]))
    elif distance == "ellipse":
        radius = get_radius(center_vector, list(vectors.values()))
        center_vector /= radius
        wv_vectors = copy(wv_model.vectors)
        wv_vectors /= radius

        point_vectors = [v/radius for v in vectors.values()]
        point_dist = cdist([center_vector], point_vectors, metric="euclidean")
        threshold = np.max(point_dist)
        all_euclidean_dist = cdist([center_vector], wv_vectors, metric="euclidean")
        for index in all_euclidean_dist.argsort()[0]:
            if all_euclidean_dist[0][index] > threshold:
                break
            closer_elements.append((index, all_euclidean_dist[0][index]))

    elif distance == "l1ellipse":
        radius = get_radius(center_vector, list(vectors.values()))
        all_l1ellipse_dist = cdist([center_vector], wv_model.vectors, metric=lambda x, y: ellipse_rectangle_dist(x, radius, y))
        all_l1ellipse_dist_1d = all_l1ellipse_dist[0]
        sorted_distances = all_l1ellipse_dist_1d.argsort()

        for index in sorted_distances:
            if all_l1ellipse_dist_1d[index] <= 1:
                closer_elements.append((index, all_l1ellipse_dist_1d[index]))
            else:
                break
    prediction = [wv_model.index_to_key[item[0]] for item in closer_elements]
    return prediction, missing

# distance_result = cdist([center_vector], [wv_model.vectors], metric=lambda x, y: is_inside_ellipse(x, radius, y))


def run_method_topk(seed, gold_size, wv_model: KeyedVectors, distance="circle", filter="url"):
    """
    Input is a single record's seed.
    distance can be the following: ["circle", "euclidean", "ellipse", "l1ellipse"]
    :return:
    """

    vectors = dict()
    missing = 0
    if filter == "url":
        filter_func = filter_url
    else:
        filter_func = filter_none

    for item in seed:
        try:
            vectors[item] = list(wv_model[item])
        except Exception as e:
            missing += 1
    if missing == len(seed):
        return None, missing
    center_vector = np.mean(list(vectors.values()), axis=0)
    closer_elements = list()

    if distance == "circle" or distance == "euclidean":
        point_dist = cdist([center_vector], list(vectors.values()), metric="euclidean")
        threshold = np.max(point_dist)
        all_euclidean_dist = cdist([center_vector], wv_model.vectors, metric="euclidean")
        for index in all_euclidean_dist.argsort()[0]:
            if not filter_func(index):
                continue
            if len(closer_elements) >= gold_size:
            # if all_euclidean_dist[0][index] > threshold or len(closer_elements) >= gold_size:
                break
            closer_elements.append((index, all_euclidean_dist[0][index]))
    elif distance == "ellipse":
        radius = get_radius(center_vector, list(vectors.values()))
        center_vector /= radius
        wv_vectors = copy(wv_model.vectors)
        wv_vectors /= radius

        point_vectors = [v/radius for v in vectors.values()]
        point_dist = cdist([center_vector], point_vectors, metric="euclidean")
        threshold = np.max(point_dist)
        all_euclidean_dist = cdist([center_vector], wv_vectors, metric="euclidean")
        for index in all_euclidean_dist.argsort()[0]:
            if not filter_func(index):
                continue
            if len(closer_elements) >= gold_size:
            # if all_euclidean_dist[0][index] > threshold or len(closer_elements) >= gold_size:
                break
            closer_elements.append((index, all_euclidean_dist[0][index]))

    elif distance == "l1ellipse":
        radius = get_radius(center_vector, list(vectors.values()))
        all_l1ellipse_dist = cdist([center_vector], wv_model.vectors, metric=lambda x, y: ellipse_rectangle_dist(x, radius, y))
        all_l1ellipse_dist_1d = all_l1ellipse_dist[0]
        sorted_distances = all_l1ellipse_dist_1d.argsort()

        for index in sorted_distances:
            if not filter_func(index):
                continue
            if len(closer_elements) >= gold_size:
                break
            closer_elements.append((index, all_l1ellipse_dist_1d[index]))
    prediction = [wv_model.index_to_key[item[0]] for item in closer_elements]
    return prediction, missing

# distance_result = cdist([center_vector], [wv_model.vectors], metric=lambda x, y: is_inside_ellipse(x, radius, y))


