from query.query_functions2 import (run_sparql_query_paged, run_sparql_query,
                                    make_triplet, count_from_triples,
                                    get_all_triplets_from_o, get_all_triplets_from_s)
import networkx as nx
from algorithm.graphwalk_functions_v4 import flattenables


def get_variables(query):
    variables = set()
    for triples in query:
        for element in triples:
            if element.startswith("?"):
                variables.add(element)
    return variables


def get_all_nodes_touched(query):
    variables = get_variables(query)

    nodes = set()
    for triples in query:
        if not triples[0].startswith("?"):
            nodes.add(triples[0].replace("<", "").replace(">", ""))

        if not triples[2].startswith("?"):
            nodes.add(triples[2].replace("<", "").replace(">", ""))
    command_string = f"SELECT DISTINCT {' '.join(variables)} WHERE {{ {' . '.join([make_triplet(item) for item in query])} }} LIMIT 5000"
    inside_nodes = run_sparql_query(
        command_string
    )

    for rec in inside_nodes:
        for vari in variables:
            nodes.add(rec[vari.replace("?", "")]["value"])

    return nodes


def get_neighbourhood(nodes):
    connections = list()

    for node in nodes:
        connections.extend(get_all_triplets_from_s(node))
        connections.extend(get_all_triplets_from_o(node))

    return connections


def get_all_connections_between_nodes(nodes_list):

    all_connections = list()
    for node in nodes_list:
        node_conns = get_all_triplets_from_s(node)
        all_connections.extend(node_conns)

    node_connections = list(filter(lambda x: x[0] in nodes_list and x[2] in nodes_list, all_connections))
    return node_connections


def get_graph_metrics(nodes, list_of_edges):
    G = nx.Graph()
    for edge in list_of_edges:
        G.add_edge(edge[0], edge[2], label=edge[1])
    diameter = nx.diameter(G)
    avg_degree = len(list_of_edges)/len(nodes)
    average_neighbor_degree = nx.average_neighbor_degree(G)
    density = nx.density(G)
    avg_path_length = nx.average_shortest_path_length(G)
    clustering_coefficient = nx.average_clustering(G)

    return diameter, avg_degree, density, avg_path_length, clustering_coefficient


def get_metrics_for_record(graph):
    nodes_touched = get_all_nodes_touched(graph)
    all_connections = get_all_connections_between_nodes(nodes_touched)
    record_metrics = get_graph_metrics(nodes_touched, all_connections)
    return record_metrics


def flatten_query_with_counts(pred_querries):
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


def query2string(triplets):
    return ' . '.join([make_triplet(item) for item in triplets]) if isinstance(triplets[0], list) else make_triplet(triplets)
