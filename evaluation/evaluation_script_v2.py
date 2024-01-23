from tqdm import tqdm
import numpy as np
from collections import defaultdict
from helper.query_helpers import flatten_query_with_counts, query2string
from evaluation.query_selectors import select_min, select_xth, select_thresholded
from query.query_functions2 import get_result_from_triples


def MAIN_graph_ranking_on_nodes(ds, pred_column="explanation", extra_info=False):
    ranks = list()
    query_sizes = list()

    for record in tqdm(ds):
        # Filter out Nones
        if isinstance(record[pred_column], list):
            continue
        if extra_info:
            record_rank, query_size_single = graph_ranking_first_nodes(record["graph"],
                                                                       record[pred_column],
                                                                       extra_info)
            query_sizes.append(query_size_single)
        else:
            record_rank = graph_ranking_first_nodes(record["graph"], record[pred_column])
        ranks.append(record_rank)

    metrics = get_mr_metrics_triples(ranks)
    if extra_info:
        return metrics, ranks, query_sizes
    return metrics


def graph_ranking_first_nodes(gold_graph, pred_graphs, extra_info=False, eps=5):
    gold_graph_with_nodes = [gold_graph, get_result_from_triples(gold_graph)]
    pred_graphs_with_counts = flatten_query_with_counts(pred_graphs)
    pred_graphs_with_counts_sorted = sorted(pred_graphs_with_counts, key=lambda x: x[1])

    rank = 0
    prev_size = -1
    found = False
    best_explanation = None
    for explanation in pred_graphs_with_counts_sorted:

        explanation_nodes = get_result_from_triples(explanation[0])
        explanation_size = len(explanation_nodes)
        if explanation_size != prev_size:
            rank += 1
            prev_size = explanation_size

        if len(gold_graph_with_nodes[1]) == explanation_size:

            if (len(set(gold_graph_with_nodes[1]) - (set(explanation_nodes))) == 0 and
                    len((set(explanation_nodes)) - set(gold_graph_with_nodes[1])) == 0):
                found = True
                best_explanation = explanation
                break

        if len(gold_graph_with_nodes[1]) + eps < explanation_size:
            rank = -1
            break

    if found is False:
        rank = -1
    if extra_info:
        return rank, pred_graphs_with_counts_sorted, best_explanation
    return rank


def MAIN_node_eval_ds(ds, filter_function, pred_column="output", gold_column="result_urlonly", top_funcs=None):
    if top_funcs is None:
        a = select_min
        b = lambda x: select_xth(x, 2)
        c = lambda x: select_thresholded(x, 100)
        top_funcs = [a, b, c]

    if "explanation" in ds[0]:
        return node_eval_ds_top3(ds, filter_function, gold_column, top_funcs)
    else:
        return node_eval_ds2(ds, filter_function, pred_column, gold_column)


def MAIN_node_eval_ds_aggregated(ds, filter_function, pred_column="output", gold_column="result_urlonly",
                                 top_funcs=None):
    res = MAIN_node_eval_ds(ds, filter_function, pred_column, gold_column, top_funcs)

    heuristic_best_counter = defaultdict(int)
    best_container = list()

    for row in res:
        best_f1 = None
        best_element = None
        best_index = None
        if len(row) == 0:
            continue
        for index, element in enumerate(row):
            if best_f1 is None:
                best_f1 = element[2]
                best_element = element
                best_index = index
                continue
            if best_f1 < element[2]:
                best_f1 = element[2]
                best_element = element
                best_index = index
        best_container.append(best_element)
        heuristic_best_counter[best_index] += 1

    prec = np.mean([item[0] for item in best_container])
    rec = np.mean([item[1] for item in best_container])
    f1 = np.mean([item[2] for item in best_container])

    output_dict = {"precision": prec, "recall": rec, "f1": f1, "best_counter": list(heuristic_best_counter.items())}
    if len(best_container[0]) == 4:
        print("OK")
        triple_match = np.mean([item[3] for item in best_container])
        output_dict["triple_match"] = triple_match

    return output_dict


def node_eval_ds2(ds, filter_function, pred_column="output", gold_column="result_urlonly"):
    result_container = list()

    for task in tqdm(ds):
        prediction = task[pred_column]
        if prediction is None:
            prediction = []
        prediction_filtered = list(filter(lambda x: filter_function(x), prediction))
        gold_filtered = list(filter(lambda x: filter_function(x), task[gold_column]))
        prec, rec, f1 = node_eval_record(gold_filtered, prediction_filtered)
        result_container.append([[prec, rec, f1]])

    return result_container


def node_eval_record(gold, prediction):
    fn = get_fn(gold, prediction)
    fp = get_fp(gold, prediction)
    tp = get_tp(gold, prediction)

    if len(tp) == 0 and len(fp) == 0:
        precision = 0
    else:
        precision = len(tp) / (len(tp) + len(fp))

    if len(tp) == 0 and len(fn) == 0:
        recall = 0
    else:
        recall = len(tp) / (len(tp) + len(fn))

    if precision == 0 or recall == 0:
        return precision, recall, 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def node_eval_ds_top3(ds, filter_function, gold_column="result_urlonly", top_funcs=[]):
    print("Node eval TOP3")
    result_container = list()

    for task in tqdm(ds):
        gold_filtered = list(filter(lambda x: filter_function(x), task[gold_column]))
        prfs = node_eval_record_top3(gold_filtered, task["explanation"], top_funcs, task["seed"], task["graph"])
        result_container.append(prfs)

    print("Node eval TOP3 Done!")
    return result_container


def node_eval_record_top3(gold, prediction, topfuncs, seed, gold_query):
    if len(prediction) == 0:
        return []

    pred_graphs_with_counts = flatten_query_with_counts(prediction)
    pred_graphs_with_counts_sorted = sorted(pred_graphs_with_counts, key=lambda x: x[1])

    container = list()

    selected_querries = list()

    for f in topfuncs:
        selected_query = f(pred_graphs_with_counts_sorted)
        if selected_query is None or query2string(selected_query[0]) in selected_querries:
            continue
        pred_nodes = get_result_from_triples(selected_query[0])
        prec, rec, f1 = node_eval_record(gold, pred_nodes)

        query_intersect = query_overlap_counter(gold_query, selected_query[0])

        container.append([prec, rec, f1, query_intersect])

        selected_querries.append(query2string(selected_query[0]))

    return container


def normalize_triple(triple):
    return [elem.replace("<", "").replace(">", "") for elem in triple]


def query_overlap_counter(gold_triples, pred_triples):
    matched_triple = 0

    for gold_triple in gold_triples:
        gold_t = normalize_triple(gold_triple)

        for candidate_triple in pred_triples:
            candidate_t = normalize_triple(candidate_triple)

            cool = True
            for i in range(len(candidate_t)):
                if gold_t[i][0] == "?" and candidate_t[i][0] == "?":
                    continue
                elif gold_t[i] != candidate_t[i]:
                    cool = False
                    break
            if cool:
                matched_triple += 1
                break

    return matched_triple / len(gold_triples)


def get_tp(gold, pred):
    tp = list()
    gold = set(gold)
    for item in pred:
        if item in gold:
            tp.append(item)
    return tp


def get_fn(gold, pred):
    """
    :return: missing gold elements from pred
    """

    fn = list()
    pred = set(pred)
    for item in gold:
        if item not in pred:
            fn.append(item)
    return fn


def get_fp(gold, pred):
    """
    :return: false pred elements
    """

    fp = list()
    gold = set(gold)
    for item in pred:
        if item not in gold:
            fp.append(item)
    return fp


def get_mr_metrics_triples(ranks, hits=None):
    if hits is None:
        hits = [1, 5, 10]

    ranks_valids = list(filter(lambda x: x != -1, ranks))
    missing = len(ranks) - len(ranks_valids)
    mrr = np.mean([1 / item for item in ranks_valids])
    mr = np.mean(ranks_valids)

    hits_metrics = list()
    for hit in hits:
        hits_metrics.append(len(list(filter(lambda x: x <= hit, ranks_valids))) / len(ranks))
    output_dict = {"MR": mr, "MRR": mrr, "rank_missing": missing}
    for i, h in enumerate(hits):
        output_dict[f"hits@{h}"] = hits_metrics[i]
    return output_dict
