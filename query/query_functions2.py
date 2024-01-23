from SPARQLWrapper import SPARQLWrapper, JSON, POST
from query.query_functions import do_excludes, do_excludes2, reltype_filter

# sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql = SPARQLWrapper("http://localhost:8890/sparql")
# "http://localhost:8890/sparql"
# "http://dbpedia.org/sparql"
sparql.setReturnFormat(JSON)


# sparql.setMethod(POST)

def run_sparql_query(query):
    sparql.setQuery(query)

    results = None
    try:
        ret = sparql.queryAndConvert()

        for r in ret["results"]["bindings"]:
            if results is None:
                results = [r]
            else:
                results.append(r)
    except Exception as e:
        print("Error -->", e)
    return results


def run_sparql_query_paged(query, max_page=5):
    results = None

    try:
        counter = 0
        res_size = 10000
        while res_size == 10000 and counter < max_page:
            # step_query = query + f" LIMIT 10000 OFFSET {counter*10000}"
            step_query = query[:query.find("{") + 1] + query + f" }} LIMIT 10000 OFFSET {counter * 10000}"
            sparql.setQuery(step_query)
            ret = sparql.queryAndConvert()

            for r in ret["results"]["bindings"]:
                if results is None:
                    results = [r]
                else:
                    results.append(r)
            res_size = len(ret["results"]["bindings"])
            counter += 1
    except Exception as e:
        print("Error -->", e)
    return results


def format_node(node):
    if node.startswith("?") or node.startswith("<"):
        return node
    elif node.startswith("http"):
        return f"<{node}>"
    elif node.startswith('"') and node.endswith('"'):
        return node
    else:
        return '"' + node + '"'


def get_all_triplets_from_s(start_node, node_excludes=[], reltype_excludes=[], paged=True, max=5):
    query_command = f"SELECT DISTINCT ?x ?y WHERE {{ {format_node(start_node)} ?x ?y . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x) DESC(?y)"
    if paged:
        res = run_sparql_query_paged(query_command, max_page=max)
    else:
        res = run_sparql_query(query_command)

    if res is None:
        return []
    return [[start_node, item["x"]["value"], item["y"]["value"]] for item in res]


def get_all_triplets_from_s_ronly(start_node, node_excludes=[], reltype_excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?x WHERE {{ {format_node(start_node)} ?x ?y . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x) DESC(?y)"
    if paged:
        res = run_sparql_query_paged(query_command)
    else:
        res = run_sparql_query(query_command)

    if res is None:
        return []
    return [item["x"]["value"] for item in res]


def get_all_triplets_from_sr(start_node, relation, excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?y WHERE {{ {format_node(start_node)} {format_node(relation)} ?y . {do_excludes2(excludes)} }}  ORDER BY DESC(?y)"
    if paged:
        res = run_sparql_query_paged(query_command)
    else:
        res = run_sparql_query(query_command)

    if res is None:
        return []
    return [[start_node, relation, item["y"]["value"]] for item in res]


def get_all_triplets_from_o(end_node, node_excludes=[], reltype_excludes=[], paged=True, max=5):
    query_command = f"SELECT DISTINCT ?x ?y WHERE {{ ?x ?y {format_node(end_node)}  . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x) DESC(?y)"
    if paged:
        res = run_sparql_query_paged(query_command, max_page=max)
    else:
        res = run_sparql_query(query_command)

    if res is None:
        return []
    return [[item["x"]["value"], item["y"]["value"], end_node] for item in res]


def get_all_triplets_from_o_ronly(end_node, node_excludes=[], reltype_excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?y WHERE {{ ?x ?y {format_node(end_node)}  . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x) DESC(?y)"
    if paged:
        res = run_sparql_query_paged(query_command)
    else:
        res = run_sparql_query(query_command)

    if res is None:
        return []
    return [item["y"]["value"] for item in res]


def get_all_triplets_from_ro(end_node, relation, excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?y WHERE {{ ?y {format_node(relation)} {format_node(end_node)} . {do_excludes2(excludes)} }} ORDER BY DESC(?y)"
    if paged:
        res = run_sparql_query_paged(query_command)
    else:
        res = run_sparql_query(query_command)

    if res is None:
        return []
    return [[item["y"]["value"], relation, end_node] for item in res]


def count_all_triplets_from_sr(start_node, relation):
    query_command = f"SELECT COUNT(?x)  WHERE {{ {format_node(start_node)} {format_node(relation)} ?x}}"
    return run_sparql_query(query_command)


def count_all_triplets_from_ro(end_node, relation):
    query_command = f"SELECT COUNT(?x)  WHERE {{ ?x {format_node(relation)} {format_node(end_node)}}}"
    return run_sparql_query(query_command)


def count_from_triples(triples):
    # query_command = f"SELECT count(?uri) WHERE {{ {' . '.join([make_triplet(item) for item in triples])} }}"
    query_command = f"SELECT count(?uri) WHERE {{ {' . '.join([make_triplet(item) for item in triples]) if isinstance(triples[0], list) else make_triplet(triples)} }}"
    query_results = run_sparql_query(query_command)
    return int(query_results[0]['callret-0']['value'])


def get_result_from_triples(triples, limit=5000):
    query_command = f"SELECT DISTINCT ?uri WHERE {{ {' . '.join([make_triplet(item) for item in triples]) if isinstance(triples[0], list) else make_triplet(triples)} . }} LIMIT {limit}"
    query_results = run_sparql_query(query_command)
    if query_results is None:
        return None
    return [item['uri']['value'] for item in query_results]


def make_triplet(triple):
    query_triplet = []
    for item in triple:
        query_triplet.append(format_node(item))
    return " ".join(query_triplet)
