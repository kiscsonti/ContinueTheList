from algorithm.kgvec2go_vectors_simple import run_method, load_gensim_model, run_method_topk
from excludes.basic_excludes import (node_excl_yago_func, node_excl_extra,
                                     node_excl_owlthing_func, node_excl_wiki_func)
from loaders.loaders import *
from evaluation.split_dataset import split_on_seed_dataset
import json
import pickle
import os
from tqdm import tqdm
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument('--metric', '-m', type=str, required=True)
parser.add_argument('--topk', '-k', type=str2bool, nargs='?', const=True, default=False)

args = parser.parse_args()

exclude_paths = [
    ["", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://www.w3.org/2004/02/skos/core#Concept"]
]
relation_excludes = ["http://xmlns.com/foaf/0.1/primaryTopic",
                     "http://dbpedia.org/property/isCitedBy",
                     "http://dbpedia.org/ontology/wikiPageWikiLink",
                     "http://dbpedia.org/ontology/wikiPageWikiLinkText",
                     "http://dbpedia.org/property/wikiPageUsesTemplate",
                     'http://dbpedia.org/ontology/wikiPageRedirects',
                     'http://dbpedia.org/ontology/wikiPageOutDegree',
                     "http://dbpedia.org/ontology/abstract",
                     "http://www.w3.org/2000/01/rdf-schema#comment", ]

res_excludes_inside_point_relation = ["http://purl.org/linguistics/gold/hypernym",
                                      "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
res_excludes_inside_point_node = []
node_excludes_funcs = [node_excl_wiki_func, node_excl_owlthing_func, node_excl_yago_func, node_excl_extra]

SEED = 44
SAMPLE_SIZE = 4
verbose = 1

#Dataset load
#lc_quad_train = load_lc_quad_train()

lc_quad_train = load_on_path(ds_lc_quad_train_cleaned)
splitted_lcquad = split_on_seed_dataset(lc_quad_train, SAMPLE_SIZE, SEED)

cwd = "/home/kardosp/continuethelist/dbpedia_vectors/kgvec2go"
distance_metric = args.metric
assert distance_metric in ["ellipse", "circle", "l1ellipse", "euclidean"]

word2vec_path = os.path.join(cwd, 'model.kv')
wv_model = load_gensim_model(word2vec_path)
missing_counter = 0

# lcquad_output = run_on_dataset(splitted_lcquad, exclude_paths, relation_excludes, node_excludes_funcs)
for i, record in tqdm(list(enumerate(splitted_lcquad))):
    # print(i)
    if "output" in record:
        continue
    if args.topk:
        result, missing = run_method_topk(record["seed"], len(record["seed"])+len(record["result"]), wv_model,
                                          distance=distance_metric)
    else:
        result, missing = run_method(record["seed"], wv_model, distance=distance_metric)
    record["output"] = result
    missing_counter += missing
    print(missing_counter)

    with open(f"/home/kardosp/continuethelist/outputs/kgvec2go_v1_output_{distance_metric}_{'topk' if args.topk else ''}_sample4.pickle", "wb") as f:
        pickle.dump(splitted_lcquad, f)

