from sentence_transformers import SentenceTransformer
import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cdist


def get_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def transform_name(name):
    name = name.replace("http://dbpedia.org/", "")
    name = name.replace("/", " - ")
    name = name.replace("_", " ")
    return name


def run_bertvectors(model):
    # TODO - list all nodes - Load/Convert-Save embeddings - Search most similar - output
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = model.encode(sentences)
    print(embeddings)


def weightedEuclidean(a, b, w):
    q = a-b
    return np.sqrt((w*q*q).sum())


def run_method_closest_n(seed, wv_model: KeyedVectors, n=10, gold_size=None):
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

    stdev = np.array(list(vectors.values())).std(axis=0)
    weight_matrix = np.zeros(len(list(vectors.values())[0]))

    for item in np.argsort(stdev)[:n]:
        weight_matrix[item] = 1

    point_dist = cdist([center_vector], list(vectors.values()),
                       metric=lambda a, b: weightedEuclidean(a, b, weight_matrix))
    threshold = np.max(point_dist)
    all_euclidean_dist = cdist([center_vector], wv_model.vectors,
                               metric=lambda a, b: weightedEuclidean(a, b, weight_matrix))
    argsort_res = all_euclidean_dist.argsort()[0]
    for index in argsort_res:
        if all_euclidean_dist[0][index] > threshold:
            break
        if gold_size is not None and len(closer_elements) >= gold_size:
            break
        closer_elements.append((index, all_euclidean_dist[0][index]))

    prediction = [wv_model.index_to_key[item[0]] for item in closer_elements]

    return prediction, missing
