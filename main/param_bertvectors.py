from algorithm.kgvec2go_vectors_simple import run_method, load_gensim_model, run_method_topk
from algorithm.bert_vectors import run_method_closest_n
from excludes.basic_excludes import (node_excl_yago_func, node_excl_extra,
                                     node_excl_owlthing_func, node_excl_wiki_func)
from loaders.loaders import *
from evaluation.split_dataset import split_on_seed_dataset

import json
import pickle
import os
from tqdm import tqdm
import argparse
import gensim.models.keyedvectors as word2vec
from time import time
from datetime import timedelta


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
parser.add_argument('--n', '-n', type=int, default=10, required=False)
parser.add_argument('--samplesize', '-sa', type=int, default=4, required=False)
parser.add_argument('--seed', '-se', type=int, default=44, required=False)

args = parser.parse_args()

SEED = args.seed
SAMPLE_SIZE = args.samplesize
verbose = 1

start_time = time()
# lc_quad_train = load_on_path(ds_lc_quad_train_cleaned)
lc_quad_all = load_on_path(ds_lc_quad_unioned_cleaned)
splitted_lcquad = split_on_seed_dataset(lc_quad_all, SAMPLE_SIZE, SEED)

#splitted_lcquad = load_on_path_pickle("/home/kardosp/continuethelist/outputs/graphwalk_v2_output_sample4.pickle")

cwd = "/home/kardosp/continuethelist/stats/extra_data"
distance_metric = args.metric
assert distance_metric in ["circle", "topk"]

output = f"/home/kardosp/continuethelist/automatic_outputs/bertvectors_v1_output_{distance_metric}_{'topk' if args.topk else ''}_{args.n}_sample{SAMPLE_SIZE}_seed{SEED}.pickle"

word2vec_path = os.path.join(cwd, 'dbpedia_gensimvectors_MiniLM.kv')
wv_model = word2vec.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
missing_counter = 0

for i, record in tqdm(list(enumerate(splitted_lcquad))):
    print(i)
    if "output" in record:
        continue
    if distance_metric == "topk":
        if args.topk:
            result, missing = run_method_closest_n(record["seed"], wv_model, args.n, len(record["result_urlonly"]))
        else:
            result, missing = run_method_closest_n(record["seed"], wv_model, args.n)
    elif distance_metric == "circle" and args.topk:
        result, missing = run_method_topk(record["seed"], len(record["result_urlonly"]), wv_model,
                                          distance=distance_metric)
    else:
        result, missing = run_method(record["seed"], wv_model, distance=distance_metric)
    record["output"] = result
    missing_counter += missing
    print(missing_counter)

    with open(output, "wb") as f:
        pickle.dump(splitted_lcquad, f)


end_time = time()
elapsed_time = end_time-start_time
with open(output.replace("outputs", "outputs_stats").replace(".pickle", ".json"), "w") as f:
    json.dump({"runtime": elapsed_time}, f)

print(timedelta(seconds=elapsed_time))
