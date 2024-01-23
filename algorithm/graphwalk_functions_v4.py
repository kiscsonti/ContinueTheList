from evaluation.evaluate_script import *
from loaders.loaders import *
from query.query_functions2 import (
    get_all_triplets_from_s, get_all_triplets_from_o,
    get_all_triplets_from_ro, get_all_triplets_from_sr,
    get_all_triplets_from_o_ronly, get_all_triplets_from_s_ronly,
    run_sparql_query_paged, run_sparql_query,
    get_result_from_triples, count_from_triples
)
from typing import List

from tqdm import tqdm
from copy import copy
import random
import pandas as pd
from IPython.display import display, HTML
import numpy as np
import json
from datetime import datetime

from collections import defaultdict

flattenables = ["forward_backward", "backward_forward", "backward_backward", "forward_forward"]


def basic_union(x, y):
    if x is None:
        return y
    if y is None:
        return x

    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def deduplicate_union(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return {k: 1 for k in set(x) | set(y)}


def get_element_from_list(triples, i=1):
    """
    In this function triples might not be triples. That is the point of the parameter: i.

    :param triples:
    :return:
    """
    for trip in triples:
        if trip[i] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            return trip
    return None


def exclude_nodes(triples, node_exclude_funcs, first=True):
    """
    :param triples:
    :param node_exclude_funcs:
    :param first: True if the triples first element shall be tested - False if the last element
    :return: list of valid triples
    """
    if node_exclude_funcs is None or len(node_exclude_funcs) == 0:
        return triples

    valid_triples = list()
    for trip in triples:
        node = trip[0] if first else trip[2]
        skip_flag = False
        for excl_func in node_exclude_funcs:
            if excl_func(node):
                skip_flag = True
                break
        if not skip_flag:
            valid_triples.append(trip)
    return valid_triples


def exclude_relations(triples, relations, relation_excludes_funcs=[]):
    valid_triples = list()

    for trip in triples:
        if trip[1] in relations:
            continue
        if any([item(trip[1]) for item in relation_excludes_funcs]):
            continue
        valid_triples.append(trip)
    return valid_triples


def exclude_relations_simple(items, relations, relation_excludes_funcs=[]):
    valid_items = list()

    for it in items:
        if it in relations:
            continue
        if any([func(it) for func in relation_excludes_funcs]):
            continue
        valid_items.append(it)
    return valid_items


def exclude_path(triples, paths):
    valid_triples = list()
    for trip in triples:
        invalid_trip = False
        for path in paths:
            flag_holds = True
            for i, path_element in enumerate(path):
                if path_element == "":
                    continue
                elif trip[i] != path_element:
                    flag_holds = False
                    break
            if flag_holds:
                invalid_trip = True
        if invalid_trip is False:
            valid_triples.append(trip)
    return valid_triples


def triples_to_dict(triples, stay: List[int] = None):
    """
    :param triples:
    :param stay: List of interegers - Determines which indexes to make the dict upon IMPORTANT: Indexes start at 0
    :return:
    """
    # Default value
    if stay is None:
        stay = [0, 1, 2]
    container = dict()

    if isinstance(stay, list) is False:
        raise TypeError("Input stay is not list of integers")

    for trip in triples:
        container["@@@".join([trip[item] for item in stay])] = 1
    return container


def get_forward1(nodes, exclude_paths=[], exclude_rels=[], node_exclude_funcs=[], relation_excludes_funcs=[]):
    overall_count_container = None

    for seed_item in nodes:
        all_triples = get_all_triplets_from_s(seed_item, exclude_paths, )

        filtered_triples = exclude_nodes(all_triples, node_exclude_funcs, first=False)
        filtered_triples = exclude_relations(filtered_triples, exclude_rels, relation_excludes_funcs)

        triples_count_container = triples_to_dict(filtered_triples, stay=[1, 2])
        overall_count_container = basic_union(overall_count_container, triples_count_container)
    top_only = list(filter(lambda x: x[1] == len(nodes), overall_count_container.items()))
    return top_only


def get_backward1(nodes, exclude_paths=[], exclude_rels=[], node_exclude_funcs=[], relation_excludes_funcs=[]):
    overall_count_container = None

    for seed_item in nodes:
        all_triples = get_all_triplets_from_o(seed_item, exclude_paths, )

        filtered_triples = exclude_nodes(all_triples, node_exclude_funcs, first=False)
        filtered_triples = exclude_relations(filtered_triples, exclude_rels, relation_excludes_funcs)

        triples_count_container = triples_to_dict(filtered_triples, stay=[0, 1])
        overall_count_container = basic_union(overall_count_container, triples_count_container)
    top_only = list(filter(lambda x: x[1] == len(nodes), overall_count_container.items()))
    return top_only


def from_insidepointbackward_moveforward(nodes, common_relations, variable_name,
                                         exclude_paths, exclude_rels, node_exclude_funcs, relation_excludes_funcs,
                                         upper_limit=2000):
    query_error = False
    full_paths = list()

    for common_rel in tqdm(common_relations, desc="commonrel"):
        q_error_flag = False
        common_relation_count_container = None
        for seed_item in nodes:
            inside_points = get_all_triplets_from_ro(seed_item, common_rel)
            if inside_points is None:
                query_error = True
                q_error_flag = True
                break
            inside_points_filtered = exclude_nodes(inside_points, node_exclude_funcs, first=True)
            inside_points_filtered = exclude_path(inside_points_filtered, exclude_paths)

            # now = datetime.now()
            # current_time = now.strftime("%d/%m/% - %H:%M:%S")
            # print(current_time, "forward_backward inside points size:", len(inside_points_filtered))
            if len(inside_points_filtered) > upper_limit:
                inside_points_filtered = random.sample(inside_points_filtered, k=upper_limit)

            inside_point_deduplicated_container = None

            for i_point in inside_points_filtered:
                ro_endpoints_raw = get_all_triplets_from_s(i_point[0])

                ro_endpoints_filtered = exclude_nodes(ro_endpoints_raw, node_exclude_funcs, first=False)
                ro_endpoints_filtered = exclude_relations(ro_endpoints_filtered, exclude_rels, relation_excludes_funcs)

                triples_count_container = triples_to_dict(ro_endpoints_filtered, stay=[1, 2])
                inside_point_deduplicated_container = deduplicate_union(inside_point_deduplicated_container,
                                                                        triples_count_container)

            common_relation_count_container = basic_union(common_relation_count_container,
                                                          inside_point_deduplicated_container)

        if q_error_flag:
            continue
        if common_relation_count_container is None:
            continue
        valid_insidepoint_paths = list(filter(lambda x: x[1] == len(nodes), common_relation_count_container.items()))
        for item in valid_insidepoint_paths:
            variable_name = next_word(variable_name)
            full_paths.append([common_rel] + ["?" + variable_name] + item[0].split("@@@"))

    return full_paths, variable_name, query_error


def from_insidepointbackward_movebackward(nodes, common_relations, variable_name, exclude_paths, exclude_rels,
                                          node_exclude_funcs, relation_excludes_funcs, upper_limit=2000):
    query_error = False
    full_paths = list()

    for common_rel in tqdm(common_relations, desc="commonrel"):
        q_error_flag = False
        common_relation_count_container = None
        for seed_item in nodes:
            inside_points = get_all_triplets_from_ro(seed_item, common_rel)
            if inside_points is None:
                query_error = True
                q_error_flag = True
                break
            inside_points_filtered = exclude_nodes(inside_points, node_exclude_funcs, first=True)
            inside_points_filtered = exclude_path(inside_points_filtered, exclude_paths)

            if len(inside_points_filtered) > upper_limit:
                inside_points_filtered = random.sample(inside_points_filtered, k=upper_limit)

            inside_point_deduplicated_container = None

            for i_point in inside_points_filtered:
                sr_endpoints_raw = get_all_triplets_from_o(i_point[0])

                sr_endpoints_filtered = exclude_nodes(sr_endpoints_raw, node_exclude_funcs, first=True)
                sr_endpoints_filtered = exclude_relations(sr_endpoints_filtered, exclude_rels, relation_excludes_funcs)

                triples_count_container = triples_to_dict(sr_endpoints_filtered, stay=[0, 1])
                inside_point_deduplicated_container = deduplicate_union(inside_point_deduplicated_container,
                                                                        triples_count_container)

            common_relation_count_container = basic_union(common_relation_count_container,
                                                          inside_point_deduplicated_container)

        if q_error_flag:
            continue
        if common_relation_count_container is None:
            continue
        valid_insidepoint_paths = list(filter(lambda x: x[1] == len(nodes), common_relation_count_container.items()))
        for item in valid_insidepoint_paths:
            variable_name = next_word(variable_name)
            full_paths.append([common_rel] + item[0].split("@@@") + ["?" + variable_name])

    return full_paths, variable_name, query_error


def from_insidepointforward_moveforward(nodes, common_relations, variable_name,
                                        exclude_paths, exclude_rels, node_exclude_funcs, relation_excludes_funcs,
                                        upper_limit=2000):
    query_error = False
    full_paths = list()

    for common_rel in tqdm(common_relations, desc="commonrel"):
        q_error_flag = False
        common_relation_count_container = None
        for seed_item in nodes:
            inside_points = get_all_triplets_from_sr(seed_item, common_rel)
            if inside_points is None:
                query_error = True
                q_error_flag = True
                break
            inside_points_filtered = exclude_nodes(inside_points, node_exclude_funcs, first=False)
            inside_points_filtered = exclude_path(inside_points_filtered, exclude_paths)

            if len(inside_points_filtered) > upper_limit:
                inside_points_filtered = random.sample(inside_points_filtered, k=upper_limit)

            inside_point_deduplicated_container = None

            for i_point in inside_points_filtered:
                ro_endpoints_raw = get_all_triplets_from_s(i_point[2])

                ro_endpoints_filtered = exclude_nodes(ro_endpoints_raw, node_exclude_funcs, first=False)
                ro_endpoints_filtered = exclude_relations(ro_endpoints_filtered, exclude_rels, relation_excludes_funcs)

                triples_count_container = triples_to_dict(ro_endpoints_filtered, stay=[1, 2])
                inside_point_deduplicated_container = deduplicate_union(inside_point_deduplicated_container,
                                                                        triples_count_container)

            common_relation_count_container = basic_union(common_relation_count_container,
                                                          inside_point_deduplicated_container)

        if q_error_flag:
            continue
        if common_relation_count_container is None:
            continue
        valid_insidepoint_paths = list(filter(lambda x: x[1] == len(nodes), common_relation_count_container.items()))
        for item in valid_insidepoint_paths:
            variable_name = next_word(variable_name)
            full_paths.append([common_rel] + ["?" + variable_name] + item[0].split("@@@"))

    return full_paths, variable_name, query_error


def from_insidepointforward_movebackward(nodes, common_relations, variable_name, exclude_paths, exclude_rels,
                                         node_exclude_funcs, relation_excludes_funcs, upper_limit=2000):
    query_error = False
    full_paths = list()

    for common_rel in tqdm(common_relations, desc="commonrel"):
        q_error_flag = False
        common_relation_count_container = None
        for seed_item in nodes:
            inside_points = get_all_triplets_from_sr(seed_item, common_rel)
            if inside_points is None:
                query_error = True
                q_error_flag = True
                break
            inside_points_filtered = exclude_nodes(inside_points, node_exclude_funcs, first=False)
            inside_points_filtered = exclude_path(inside_points_filtered, exclude_paths)

            # now = datetime.now()
            # current_time = now.strftime("%d/%m/% - %H:%M:%S")
            # print(current_time, "forward_backward inside points size:", len(inside_points_filtered))
            if len(inside_points_filtered) > upper_limit:
                inside_points_filtered = random.sample(inside_points_filtered, k=upper_limit)

            inside_point_deduplicated_container = None

            for i_point in inside_points_filtered:
                sr_endpoints_raw = get_all_triplets_from_o(i_point[2])

                sr_endpoints_filtered = exclude_nodes(sr_endpoints_raw, node_exclude_funcs, first=True)
                sr_endpoints_filtered = exclude_relations(sr_endpoints_filtered, exclude_rels, relation_excludes_funcs)

                triples_count_container = triples_to_dict(sr_endpoints_filtered, stay=[0, 1])
                inside_point_deduplicated_container = deduplicate_union(inside_point_deduplicated_container,
                                                                        triples_count_container)

            common_relation_count_container = basic_union(common_relation_count_container,
                                                          inside_point_deduplicated_container)

        if q_error_flag:
            continue
        if common_relation_count_container is None:
            continue
        valid_insidepoint_paths = list(filter(lambda x: x[1] == len(nodes), common_relation_count_container.items()))
        for item in valid_insidepoint_paths:
            variable_name = next_word(variable_name)
            full_paths.append([common_rel] + item[0].split("@@@") + ["?" + variable_name])

    return full_paths, variable_name, query_error


def get_backward_insidepoint(nodes, variable_state, res_backward, exclude_paths=[], exclude_rels=[],
                             node_exclude_funcs=[],
                             backward_step1_rel_excludes=[], relation_excludes_funcs=[]):
    relation_count_container = None

    backward_rels = [item[1] for item in res_backward]
    for seed_item in nodes:
        all_relations = get_all_triplets_from_o_ronly(seed_item, exclude_paths, )
        # all_relations = list(filter(lambda x: x not in exclude_rels, all_relations))
        all_relations = exclude_relations_simple(all_relations,
                                                 exclude_rels + backward_step1_rel_excludes + backward_rels,
                                                 relation_excludes_funcs)
        # To dict
        triples_count_container = {item: 1 for item in all_relations}
        relation_count_container = basic_union(relation_count_container, triples_count_container)
    common_relations_tuple = list(filter(lambda x: x[1] == len(nodes), relation_count_container.items()))
    common_relations = [item[0] for item in common_relations_tuple]

    # with open("/home/kardosp/continuethelist/notebooks/logs/common_rels_backward.json", "r") as f:
    #     backward_rels = json.load(f)
    #
    # backward_rels.extend(common_relations)
    # with open("/home/kardosp/continuethelist/notebooks/logs/common_rels_backward.json", "w") as f:
    #     json.dump(backward_rels, f)
    print("backward-forward")
    variable_state2 = variable_state
    full_paths_forward, variable_state2, query_error1 = from_insidepointbackward_moveforward(nodes,
                                                                                             common_relations,
                                                                                             variable_state2,
                                                                                             exclude_paths,
                                                                                             exclude_rels,
                                                                                             node_exclude_funcs,
                                                                                             relation_excludes_funcs)

    print("backward-backward")
    full_paths_backward, variable_state2, query_error2 = from_insidepointbackward_movebackward(nodes,
                                                                                               common_relations,
                                                                                               variable_state2,
                                                                                               exclude_paths,
                                                                                               exclude_rels,
                                                                                               node_exclude_funcs,
                                                                                               relation_excludes_funcs)

    # query_error = any([query_error1, query_error2])
    insidepoint_forward = list()
    insidepoint_backward = list()
    for item in full_paths_forward:
        insidepoint_forward.append([[item[1], item[0], "?uri"], [item[1], item[2], item[3]]])

    for item in full_paths_backward:
        insidepoint_backward.append([[item[3], item[0], "?uri"], [item[1], item[2], item[3]]])

    # return insidepoint_forward, variable_state2, query_error1
    return insidepoint_forward, insidepoint_backward, variable_state2, query_error1, query_error2


def get_forward_insidepoint(nodes, variable_state, res_forward, exclude_paths=[], exclude_rels=[],
                            node_exclude_funcs=[],
                            step1_rel_excludes=[], relation_excludes_funcs=[]):
    relation_count_container = None

    forward_rels = [item[1] for item in res_forward]
    for seed_item in nodes:
        all_relations = get_all_triplets_from_s_ronly(seed_item, exclude_paths, )
        # all_relations = list(filter(lambda x: x not in exclude_rels, all_relations))
        all_relations = exclude_relations_simple(all_relations, exclude_rels + forward_rels, relation_excludes_funcs)
        # To dict
        triples_count_container = {item: 1 for item in all_relations}
        relation_count_container = basic_union(relation_count_container, triples_count_container)
    common_relations_tuple = list(filter(lambda x: x[1] == len(nodes), relation_count_container.items()))
    common_relations = [item[0] for item in common_relations_tuple]
    common_relations = list(filter(lambda x: x not in step1_rel_excludes, common_relations))

    # with open("/home/kardosp/continuethelist/notebooks/logs/common_rels_forward.json", "r") as f:
    #     forward_rels = json.load(f)
    #
    # forward_rels.extend(common_relations)
    # with open("/home/kardosp/continuethelist/notebooks/logs/common_rels_forward.json", "w") as f:
    #     json.dump(forward_rels, f)
    variable_state2 = variable_state
    full_paths_forward, variable_state2, query_error1 = from_insidepointforward_moveforward(nodes,
                                                                                            common_relations,
                                                                                            variable_state2,
                                                                                            exclude_paths,
                                                                                            exclude_rels,
                                                                                            node_exclude_funcs,
                                                                                            relation_excludes_funcs)

    full_paths_backward, variable_state2, query_error2 = from_insidepointforward_movebackward(nodes,
                                                                                              common_relations,
                                                                                              variable_state2,
                                                                                              exclude_paths,
                                                                                              exclude_rels,
                                                                                              node_exclude_funcs,
                                                                                              relation_excludes_funcs)

    # query_error = any([query_error1, query_error2])
    insidepoint_forward = list()
    insidepoint_backward = list()
    for item in full_paths_forward:
        insidepoint_forward.append([["?uri", item[0], item[1]], [item[1], item[2], item[3]]])

    for item in full_paths_backward:
        insidepoint_backward.append([["?uri", item[0], item[3]], [item[1], item[2], item[3]]])

    # return insidepoint_backward, variable_state2, query_error2
    return insidepoint_forward, insidepoint_backward, variable_state2, query_error1, query_error2


def show_insidepoint_results(result, excludes_node, excludes_rel):
    showable = list()
    for item in result:
        if item[-1] in excludes_node:
            continue
        if item[-2] in excludes_rel:
            continue
        showable.append(item)
    return showable


variable_start = "aaaaa"
invalid_variable = ["uri"]


def next_word(word_state):
    i = 1
    new_word = ""
    done = False
    while not done:

        x = ord(word_state[-i])
        x = x + 1
        if x == 123:
            new_word = "a" + new_word
            i += 1
        else:
            new_word = word_state[:i] + chr(x) + new_word
            done = True

    if new_word in invalid_variable:
        return next_word(new_word)
    return new_word


def get_task_solution(task_seed, exclude_paths=[], relation_excludes=[], node_excludes_funcs=[],
                      forward_step1_rel_excludes=[], backward_step1_rel_excludes=[], relation_excludes_funcs=[]):
    variable_state = variable_start
    final_connection_list = defaultdict(list)
    res_forward = get_forward1(task_seed, exclude_paths, relation_excludes, node_excludes_funcs,
                               relation_excludes_funcs)
    for item in res_forward:
        conn = item[0].split("@@@")
        final_connection_list["forward"].append(["?uri"] + conn)
    print("Forward Done!")

    res_backward = get_backward1(task_seed, exclude_paths, relation_excludes, node_excludes_funcs,
                                 relation_excludes_funcs)
    for item in res_backward:
        conn = item[0].split("@@@")
        final_connection_list["backward"].append(conn + ["?uri"])
    print("Backward Done!")
    # (res_backward_forward, variable_state, qerror_bf)
    (res_backward_forward, res_backward_backward,
     variable_state,
     bf_error, bb_error) = get_backward_insidepoint(task_seed, variable_state,
                                                    final_connection_list["backward"],
                                                    exclude_paths,
                                                    relation_excludes,
                                                    node_excludes_funcs,
                                                    backward_step1_rel_excludes,
                                                    relation_excludes_funcs,
                                                    )
    for item in res_backward_forward:
        final_connection_list["backward_forward"].append(item)
    print("Backward_forward Done!")
    for item in res_backward_backward:
        final_connection_list["backward_backward"].append(item)
    print("backward_backward Done!")

    # (res_forward_backward, variable_state, qerror_fb)
    (res_forward_forward, res_forward_backward,
     variable_state,
     ff_error, fb_error) = get_forward_insidepoint(task_seed, variable_state,
                                                   final_connection_list["forward"],
                                                   exclude_paths, relation_excludes,
                                                   node_excludes_funcs,
                                                   forward_step1_rel_excludes,
                                                   relation_excludes_funcs,
                                                   )
    for item in res_forward_forward:
        final_connection_list["forward_forward"].append(item)
    print("forward_forward Done!")
    for item in res_forward_backward:
        final_connection_list["forward_backward"].append(item)
    print("Forward_backward Done!")

    query_error = any([ff_error, fb_error, bb_error, bf_error])

    print("res_forward", len(res_forward))
    print("res_backward", len(res_backward))
    print("backward_forward", len(res_backward_forward))
    print("backward_backward", len(res_backward_backward))
    print("forward_backward", len(res_forward_backward))
    print("forward_forward", len(res_forward_forward))
    print("had error:", query_error)
    # print("backward_backward", len(res_backward_backward))
    # print("forward_forward", len(res_forward_forward))
    # print("forward_backward", len(res_forward_forward))
    return final_connection_list


def make_triplet(triple):
    query_triplet = []
    for item in triple:
        if item.startswith("?"):
            query_triplet.append(item)
        elif item.startswith("http"):
            query_triplet.append(f"<{item}>")
        else:
            query_triplet.append('"' + item + '"')
    return " ".join(query_triplet)


def clear_results(results, verbose=0):
    """
    The point of this function is to run the querries one by one. If one of the querries doesn't work (returns None) it would be discarded from the resulting querries.
    Prevents faulty querries.
    :param results:
    :return:
    """
    valid_querries = list()
    invalid_querries = list()
    for res in tqdm(results):
        query_res = get_result_from_triples(res, 10)
        if query_res is not None:
            valid_querries.append(res)
        else:
            invalid_querries.append(res)
    return valid_querries, invalid_querries


def flatten_result(dict_type_results):
    resulting_triples = list()
    for k, v in dict_type_results.items():
        if k in flattenables:
            for item in v:
                for triplet in item:
                    # print("V:", v)
                    # print("item:", item)
                    resulting_triples.append(triplet)
        else:
            for item in v:
                resulting_triples.append(item)
    return resulting_triples


def update_resultset(results, core_query, result_set=None, key="backward_forward", verbose=0):
    if verbose > 0:
        gen = tqdm(results[key])
    else:
        gen = results[key]

    for back2, cnt in gen:
        core_q = copy(core_query)

        for triple in back2:
            core_q.append(triple)

        query_res = set(get_result_from_triples(core_q))

        if result_set is None:
            result_set = query_res
        else:
            result_set = result_set.intersection(query_res)

    return result_set


def complex_partly_query_run(results: dict, upperlimit=50, verbose=0):
    """
    Given a dict of resulting triples separated into template groups.
    Sometimes there are too much triples to be forwarded to query, so it is separated into multiple ones.
    :param results:
    :return: Runs separated querries with the resulting sets intersected so that the overall node set is returned
    """
    print("Complex")
    core_query = results["forward"] + results["backward"]

    if len(results["backward_forward"]) > 0:
        result_set = update_resultset(results, core_query, None, "backward_forward", verbose=verbose)
    else:
        result_set = set(get_result_from_triples(core_query))

    if len(results["forward_backward"]) > 0:
        result_set = update_resultset(results, core_query, result_set, "forward_backward", verbose)

    return result_set


def complex_partly_query_run2(results: dict, verbose=0):
    """
    Given a dict of resulting triples separated into template groups.
    Sometimes there are too much triples to be forwarded to query, so it is separated into multiple ones.
    :param results:
    :return: Runs separated querries with the resulting sets intersected so that the overall node set is returned
    """
    print("Complex2")

    smallest_query = None
    smallest_size = None

    for k, v in results.items():
        if k == "invalids":
            continue
        for query in v:
            if k in flattenables:
                q = query[0]
            else:
                q = query
            size = count_from_triples(q)
            if smallest_size is None:
                smallest_size = size
                smallest_query = q
            elif smallest_size > size:
                smallest_size = size
                smallest_query = q
    return set(get_result_from_triples(smallest_query))


def run_grapwalk_function_v4(record_seed, exclude_paths, relation_excludes, node_excludes_funcs,
                             forward_step1_rel_excludes, backward_step1_rel_excludes,
                             relation_excludes_funcs, verbose=0):
    """
    Runs the whole algorithm of graph walking
    :param record_seed:
    :param exclude_paths:
    :param relation_excludes:
    :param node_excludes_funcs:
    :return: [extension of input seed, explanation in form of graph triples]
    """
    result = get_task_solution(record_seed, exclude_paths, relation_excludes, node_excludes_funcs,
                               forward_step1_rel_excludes, backward_step1_rel_excludes, relation_excludes_funcs)
    filtered_results = defaultdict(list)
    for k, v in result.items():
        valid, invalid = clear_results(v)
        filtered_results[k] = valid
        filtered_results["invalids"].extend(invalid)

    backward_forward = list(map(lambda x: [x, count_from_triples(x)], filtered_results["backward_forward"]))
    filtered_results["backward_forward"] = sorted(backward_forward, key=lambda x: x[1])

    forward_backward = list(map(lambda x: [x, count_from_triples(x)], filtered_results["forward_backward"]))
    filtered_results["forward_backward"] = sorted(forward_backward, key=lambda x: x[1])

    backward_forward = list(map(lambda x: [x, count_from_triples(x)], filtered_results["backward_backward"]))
    filtered_results["backward_backward"] = sorted(backward_forward, key=lambda x: x[1])

    forward_backward = list(map(lambda x: [x, count_from_triples(x)], filtered_results["forward_forward"]))
    filtered_results["forward_forward"] = sorted(forward_backward, key=lambda x: x[1])

    triples_size = (len(filtered_results["forward"]) +
                    len(filtered_results["backward"]) +
                    len(filtered_results["backward_forward"]) +
                    len(filtered_results["forward_backward"]) +
                    len(filtered_results["backward_backward"]) +
                    len(filtered_results["forward_forward"])
                    )
    if triples_size == 0:
        return [], []

    record_results = complex_partly_query_run2(filtered_results)

    if verbose > 0:
        print("record_seed", record_seed)
        print("record_results: ", record_results)

        for k in result.keys():
            print(k, len(result[k]), len(filtered_results[k]))
        if record_results is not None:
            print("len record_results: ", len(record_results))

        # print("Invalids:")
        # print(filtered_results["invalids"])

    return record_results, filtered_results


def run_on_dataset(dataset, exclude_paths, relation_excludes, node_excludes_funcs,
                   forward_step1_rel_excludes, backward_step1_rel_excludes,
                   relation_excludes_funcs, verbose=0):
    dataset2 = copy(dataset)
    for record in dataset2:
        result, explanation = run_grapwalk_function_v4(record["seed"], exclude_paths, relation_excludes,
                                                       node_excludes_funcs,
                                                       forward_step1_rel_excludes,
                                                       backward_step1_rel_excludes,
                                                       relation_excludes_funcs,
                                                       verbose=verbose)
        record["output"] = result
        record["explanation"] = explanation
    return dataset2
