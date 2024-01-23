import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy, deepcopy
from query.query_functions2 import count_from_triples, get_result_from_triples
import os
from helper.query_helpers import get_metrics_for_record
from algorithm.graphwalk_functions_v4 import flattenables


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


@DeprecationWarning
def node_eval_ds(ds, filter_function, pred_column="output"):
    print("OK")
    precision_container = list()
    recall_container = list()
    f1_container = list()

    for task in tqdm(ds):
        prediction = task[pred_column]
        if prediction is None:
            prediction = []
        prediction_filtered = list(filter(lambda x: filter_function(x), prediction))
        prec, rec, f1 = node_eval_record(task["result_urlonly"], prediction_filtered)
        precision_container.append(prec)
        recall_container.append(rec)
        f1_container.append(f1)

    return np.mean(precision_container), np.mean(recall_container), np.mean(f1_container)


def node_eval_ds2(ds, filter_function, pred_column="output", gold_column="result_urlonly"):
    result_container = list()

    for task in tqdm(ds):
        prediction = task[pred_column]
        if prediction is None:
            prediction = []
        prediction_filtered = list(filter(lambda x: filter_function(x), prediction))
        prec, rec, f1 = node_eval_record(task[gold_column], prediction_filtered)
        result_container.append([[prec, rec, f1]])

    return result_container


def node_eval_ds_top3(ds, filter_function, pred_column="output", gold_column="result_urlonly", top_funcs=[]):
    print("Node eval TOP3")
    result_container = list()

    for task in tqdm(ds):
        prediction = task[pred_column]
        if prediction is None:
            prediction = []
        prediction_filtered = list(filter(lambda x: filter_function(x), prediction))
        prfs = node_eval_record_top3(task[gold_column], prediction_filtered, top_funcs)

    print("Node eval TOP3 Done!")
    return result_container


def node_eval_record_top3(gold, prediction, topfuncs):
    pred_graphs_with_counts = get_query_counts(prediction)
    pred_graphs_with_counts_sorted = sorted(pred_graphs_with_counts, key=lambda x: x[1])
    # TODO
    pass


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


def node_eval_ds_no_seed(ds, filter_function, pred_column="output"):
    precision_container = list()
    recall_container = list()
    f1_container = list()

    for task in tqdm(ds):

        given_elements = task["seed"]
        gold_elements = task["gold"]
        prediction_elements = task[pred_column]
        if prediction_elements is None:
            prediction_elements = []

        gold_elements = list(set(gold_elements) - set(given_elements))
        prediction_elements = list(set(prediction_elements) - set(given_elements))

        prediction_filtered = list(filter(lambda x: filter_function(x), prediction_elements))
        prec, rec, f1 = node_eval_record(gold_elements, prediction_filtered)
        precision_container.append(prec)
        recall_container.append(rec)
        f1_container.append(f1)

    return np.mean(precision_container), np.mean(recall_container), np.mean(f1_container)


def normalize_triple(triple, add_sign=True):
    new_triple = list()
    for element in triple:
        if element.startswith("?"):
            new_triple.append("x")
        elif add_sign and "http" in element:
            new_triple.append("<" + element + ">")
        else:
            new_triple.append(element)

    return new_triple


def graph_eval_relaxed(graph, explanation):
    remaining_graph = copy(graph)
    remaining_graph_normalized = [normalize_triple(triple, False) for triple in remaining_graph]
    remaining_graph_normalized_as_string = ["#".join(triple) for triple in remaining_graph_normalized]

    for triple in explanation:
        triple_normalized = normalize_triple(triple, True)
        triple_normalized_as_string = "#".join(triple_normalized)
        next_remaining_graph_normalized_as_string = list()
        for item in remaining_graph_normalized_as_string:
            if item != triple_normalized_as_string:
                next_remaining_graph_normalized_as_string.append(item)

        remaining_graph_normalized_as_string = next_remaining_graph_normalized_as_string

        if len(remaining_graph_normalized_as_string) == 0:
            break

    if len(remaining_graph_normalized_as_string) == 0:
        return 1
    return 0


def unpack_explanation(explanation):
    triples = list()

    for k, v in explanation.items():
        if k == "invalids":
            continue
        if k in flattenables:
            for item in v:
                triples.extend(item[0])
        else:
            triples.extend(v)
    return triples


def graph_eval_relaxed_ds(ds, pred_column="explanation", extra_info=False):
    correct = 0
    extra = list()
    for record in tqdm(ds):
        # Filter out Nones
        if isinstance(record[pred_column], list):
            continue
        explanation = unpack_explanation(record[pred_column])
        graph_score = graph_eval_relaxed(record["graph"], explanation)

        extra.append(graph_score)
        if graph_score == 1:
            correct += 1

    if extra_info:
        return correct / len(ds), extra
    return correct / len(ds)


def get_query_counts(pred_querries):
    querries_with_count = list()

    for k, v in pred_querries.items():
        if k == "invalids":
            continue
        elif k in flattenables:
            querries_with_count.extend(v)
        else:
            if len(v) == 0:
                continue
            for simple_query in v:
                querries_with_count.append([[simple_query], count_from_triples(simple_query)])
    return querries_with_count


def query_overlap(all_triples, smaller):
    smaller_bools = list()

    for triple in smaller:
        less_go = False
        for rem in all_triples:
            cool = True
            for i in range(len(triple)):
                rem = [elem.replace("<", "").replace(">", "") for elem in rem]
                if triple[i][0] == "?" and rem[i][0] == "?":
                    continue
                if triple[i] != rem[i]:
                    cool = False
                    break
            if cool:
                less_go = True
                break
        smaller_bools.append(less_go)

    return all(smaller_bools)


def graph_eval_ranking_first(gold_querry, pred_querries):
    querries_with_count_sorted = sorted(pred_querries, key=lambda x: x[1])
    querries_with_count_sorted_selection = [item[0] for item in querries_with_count_sorted]
    rank_counter = 0
    for i, element in enumerate(querries_with_count_sorted):
        if i == 0:
            rank_counter += 1
        elif querries_with_count_sorted[i - 1][1] != element[1]:
            rank_counter += 1

        overlap = query_overlap(gold_querry, querries_with_count_sorted_selection[i])
        if overlap:
            return rank_counter

    return -1


def get_mr_metrics_triples(ranks, hits=[1, 5, 10]):
    ranks_valids = list(filter(lambda x: x != -1, ranks))
    missing = len(ranks) - len(ranks_valids)
    mrr = np.mean([1 / item for item in ranks_valids])
    mr = np.mean(ranks_valids)

    hits_metrics = list()
    for hit in hits:
        hits_metrics.append(len(list(filter(lambda x: x <= hit, ranks_valids))) / len(ranks))

    return mr, mrr, hits_metrics, missing


def graph_eval_ranking_ds_triples(ds, pred_column="explanation", recalculate=True, extra_info=False):
    ranks = list()

    for record in tqdm(ds):
        # Filter out Nones
        if isinstance(record[pred_column], list):
            continue
        if recalculate:
            querries_with_count = get_query_counts(record[pred_column])
        else:
            querries_with_count = record[pred_column]

        record_rank = graph_eval_ranking_first(record["graph"], querries_with_count)
        ranks.append(record_rank)

    metrics = get_mr_metrics_triples(ranks, )
    if extra_info:
        return metrics, ranks
    return metrics


def graph_eval_ranking_first_nodes(gold_graph, pred_graphs, extra_info=False, eps=5):
    gold_graph_with_nodes = [gold_graph, get_result_from_triples(gold_graph)]
    pred_graphs_with_counts = get_query_counts(pred_graphs)
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


def get_result_group(group_rank, pred_graphs):
    pred_graphs_with_counts = get_query_counts(pred_graphs)
    pred_graphs_with_counts_sorted = sorted(pred_graphs_with_counts, key=lambda x: x[1])

    rank = 0
    prev_size = -1
    explanation_group = list()
    for explanation in pred_graphs_with_counts_sorted:

        explanation_nodes = get_result_from_triples(explanation[0])
        explanation_size = len(explanation_nodes)
        if explanation_size != prev_size:
            rank += 1
            prev_size = explanation_size

        if rank >= group_rank:
            break

        if rank == group_rank:
            explanation_group.append(explanation[0])

    return explanation_group


def get_pre_result_group(group_rank, pred_graphs):
    pred_graphs_with_counts = get_query_counts(pred_graphs)
    pred_graphs_with_counts_sorted = sorted(pred_graphs_with_counts, key=lambda x: x[1])

    rank = 0
    prev_size = -1
    explanation_group = list()
    for explanation in pred_graphs_with_counts_sorted:

        explanation_nodes = get_result_from_triples(explanation[0])
        explanation_size = len(explanation_nodes)
        if explanation_size != prev_size:
            rank += 1
            prev_size = explanation_size

        if rank > group_rank:
            break

        if rank <= group_rank:
            explanation_group.append(explanation[0])

    return explanation_group


def graph_eval_ranking_ds_nodes(ds, pred_column="explanation", extra_info=False):
    ranks = list()
    query_sizes = list()

    for record in tqdm(ds):
        # Filter out Nones
        if isinstance(record[pred_column], list):
            continue
        if extra_info:
            record_rank, query_size_single = graph_eval_ranking_first_nodes(record["graph"],
                                                                            record[pred_column],
                                                                            extra_info)
            query_sizes.append(query_size_single)
        else:
            record_rank = graph_eval_ranking_first_nodes(record["graph"], record[pred_column])
        ranks.append(record_rank)

    metrics = get_mr_metrics_triples(ranks)
    if extra_info:
        return metrics, ranks, query_sizes
    return metrics


def graph_eval_readable_ranking_main(ds, output_file, pred_column="explanation", max_col=5):
    data_matrix = list()

    for record in tqdm(ds):
        # Filter out Nones
        if isinstance(record[pred_column], list):
            continue
        gold_graph = record["graph"]
        gold_graph_with_nodes = [gold_graph, get_result_from_triples(gold_graph)]
        pred_graphs_with_counts = get_query_counts(record[pred_column])
        pred_graphs_with_counts_sorted = sorted(pred_graphs_with_counts, key=lambda x: x[1])

        rank = 0
        prev_size = -1
        found = False
        prediction_valid_graph = None
        prediction_valid_graph_index = None
        for i, explanation in enumerate(pred_graphs_with_counts_sorted):

            explanation_nodes = get_result_from_triples(explanation[0])
            explanation_size = len(explanation_nodes)
            if explanation_size != prev_size:
                rank += 1
                prev_size = explanation_size

            if len(gold_graph_with_nodes[1]) == explanation_size:

                if (len(set(gold_graph_with_nodes[1]) - (set(explanation_nodes))) == 0 and
                        len((set(explanation_nodes)) - set(gold_graph_with_nodes[1])) == 0):
                    found = True
                    prediction_valid_graph = explanation[0]
                    prediction_valid_graph_index = i
                    break

                if i >= max_col and len(gold_graph_with_nodes[1]) > explanation_size:
                    rank = -1
                    break

        if found is False:
            rank = -1

        data_record = list()
        data_record.append([gold_graph, len(gold_graph_with_nodes[1])])
        if prediction_valid_graph_index is None:
            for i in range(max_col + 2):
                data_record.append("")
        else:
            for i in range(max_col):
                if i >= prediction_valid_graph_index:
                    data_record.append("")
                else:
                    data_record.append(pred_graphs_with_counts_sorted[i])
            data_record.append(pred_graphs_with_counts_sorted[prediction_valid_graph_index][0])
            data_record.append(rank)
        data_matrix.append(data_record)

    pdf = pd.DataFrame(data_matrix)
    pdf.columns = ["gold"] + [f"pred_{i}" for i in range(max_col)] + ["match_graph", "match_rank"]
    pdf.to_excel(output_file)

    data_matrix_cleaned = list()

    for row in data_matrix:
        modified_row = list()
        for i, cell in enumerate(row):
            querries = list()
            if not isinstance(cell, list):
                modified_row.append(cell)
            else:
                if i == max_col + 1:
                    for triple in cell:
                        new_triple = [element.replace("<http://dbpedia.org/", "").replace(">", "")
                                      for element in triple]
                        querries.append(new_triple)
                    modified_row.append(querries)
                else:
                    for triple in cell[0]:
                        new_triple = [element.replace("<http://dbpedia.org/", "").replace(">", "")
                                      for element in triple]
                        querries.append(new_triple)
                    modified_row.append([querries, cell[1]])
        data_matrix_cleaned.append(modified_row)

    pdf_cleaned = pd.DataFrame(data_matrix_cleaned)
    pdf_cleaned.columns = ["gold"] + [f"pred_{i}" for i in range(max_col)] + ["match_graph", "match_rank"]
    pdf_cleaned.to_excel(output_file.replace(".xlsx", "_cleaned.xlsx"))

    return pdf_cleaned


def graph_eval_readable_ranking_comparable_main(ds, output_folder, pred_column="explanation", max_col=5):
    j = 0
    for record in tqdm(ds):
        j += 1
        data_matrix = list()
        gold_column = list()
        # Filter out Nones
        if isinstance(record[pred_column], list):
            continue
        gold_graph = record["graph"]
        gold_graph_with_nodes = [gold_graph, get_result_from_triples(gold_graph)]
        pred_graphs_with_counts = get_query_counts(record[pred_column])
        pred_graphs_with_counts_sorted = sorted(pred_graphs_with_counts, key=lambda x: x[1])

        gold_column.append(gold_graph)
        gold_column.extend(gold_graph_with_nodes[1])
        data_matrix.append(gold_column)

        miss_columns = list()
        match_column = list()

        rank = 0
        prev_size = -1
        found = False
        prediction_valid_graph = None
        prediction_valid_graph_index = None
        for i, explanation in enumerate(pred_graphs_with_counts_sorted):
            explanation_nodes = get_result_from_triples(explanation[0])
            explanation_size = len(explanation_nodes)
            if explanation_size != prev_size:
                rank += 1
                prev_size = explanation_size

            if len(gold_graph_with_nodes[1]) == explanation_size:

                if (len(set(gold_graph_with_nodes[1]) - (set(explanation_nodes))) == 0 and
                        len((set(explanation_nodes)) - set(gold_graph_with_nodes[1])) == 0):
                    found = True
                    prediction_valid_graph = explanation[0]
                    prediction_valid_graph_index = i

            if len(miss_columns) >= max_col and len(gold_graph_with_nodes[1]) > explanation_size:
                rank = -1
                break

            if found:
                match_column.append(explanation[0])
                match_column.extend(explanation_nodes)
                break
            else:
                if len(miss_columns) < max_col:
                    miss_column = list()
                    miss_column.append(explanation[0])
                    miss_column.extend(explanation_nodes)

                    miss_columns.append(miss_column)

        if found is False:
            rank = -1
            match_column = ["MISS"]

        data_matrix.extend(miss_columns)

        while len(data_matrix) != max_col + 1:
            data_matrix.append([])
        data_matrix.append(match_column)

        # Extend list to longest length (fill with empty string)
        max_column_length = len(max(data_matrix, key=lambda x: len(x)))
        for i in range(len(data_matrix)):
            while len(data_matrix[i]) != max_column_length:
                data_matrix[i].append("")

        pdf = pd.DataFrame(data_matrix).transpose()
        pdf.columns = ["gold_nodes"] + [f"pred_{i}" for i in range(5)] + ["match_nodes"]
        gold_nodes = pdf["gold_nodes"].tolist()[1:]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        outpath = os.path.join(output_folder, str(j) + ".xlsx")
        # pdf = (pdf.style
        #        .apply(lambda x: ["background: green" if v in gold_nodes else "" for v in x], axis=1)
        #        .to_excel(outpath))

        # TODO - https://stackoverflow.com/questions/73187140/how-to-export-excel-from-pandas-with-custom-style-alternate-row-colors-xlsxw
        writer = pd.ExcelWriter(outpath, engine='xlsxwriter')
        pdf.to_excel(writer, sheet_name='Sheet1', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        main_format = workbook.add_format({'bg_color': '#00B050'})
        worksheet.conditional_format(f'B3:H{max_column_length + 1}',
                                     {'type': 'formula',
                                      'criteria': f'=NOT(ISNA(VLOOKUP(B3,$A$3:H{max_column_length},1,FALSE)))',
                                      'format': main_format})

        writer.close()


def touched_graph_ranking(ds, gold=None, pred_column="explanation"):
    if gold is None:
        gold = []
        for record in ds:
            gold.append(get_metrics_for_record(record["graph"]))
        print("Gold Done!")

    for i, record in enumerate(ds):
        record_rank, query_size_single, best_explanation = graph_eval_ranking_first_nodes(record["graph"],
                                                                                          record[pred_column],
                                                                                          True)
        explanation_group = get_pre_result_group(record_rank, record[pred_column])

        for explanation in explanation_group:
            expl_metrics = get_metrics_for_record(explanation)
        # Nézzük meg hogy a best_explanation hanyadik helyen áll az expl_metrics különböző elemeit nézve a goldhoz hasonlítva.


