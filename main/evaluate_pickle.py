import argparse
import pickle
from excludes.filters import filter_url, filter_none
from evaluation.evaluation_script_v2 import MAIN_node_eval_ds_aggregated, MAIN_graph_ranking_on_nodes
from evaluation.query_selectors import select_min, select_xth, select_thresholded, select_average
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--file', '-f', type=str, required=True)
parser.add_argument('--output', '-o', type=str, default="/home/kardosp/continuethelist/evals")
parser.add_argument('--filter', '-fi', type=str, default="url")

args = parser.parse_args()

a = select_min
# b = lambda x: select_xth(x, 2)
b = lambda x: select_average(x, 100)
c = lambda x: select_thresholded(x, 100)
top_funcs = [a, b, c]


with open(args.file, "rb") as f:
    algo_results = pickle.load(f)

if args.filter == "url":
    filter_function = filter_url
else:
    filter_function = filter_none


output_dict = MAIN_node_eval_ds_aggregated(algo_results, filter_function, pred_column="output",
                                           gold_column="result_urlonly", top_funcs=top_funcs)

if "explanation" in algo_results[0]:
    rank_metrics = MAIN_graph_ranking_on_nodes(algo_results, "explanation", False)

    for k, v in rank_metrics.items():
        output_dict[k] = v

file_name = os.path.split(args.file)[-1].split(".")[0] + ".json"

with open(os.path.join(args.output, file_name), "w") as of:
    json.dump(output_dict, of, indent=4)
