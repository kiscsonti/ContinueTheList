from SPARQLWrapper import SPARQLWrapper, JSON, POST

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


def run_sparql_query_paged(query):
    results = None

    try:
        counter = 0
        res_size = 10000
        while res_size == 10000 and counter < 5:
            # step_query = query + f" LIMIT 10000 OFFSET {counter*10000}"
            step_query = query[:query.find("{") + 1] + query + f" }} LIMIT 10000 OFFSET {counter * 10000}"
            # print(f"Query counter: {counter}")
            # print(f"Query: {step_query}")
            # print(f"res_size: {res_size}")
            sparql.setQuery(step_query)
            ret = sparql.queryAndConvert()

            for r in ret["results"]["bindings"]:
                if results is None:
                    results = [r]
                else:
                    results.append(r)
            # print("return size:",  len(ret["results"]["bindings"]))
            res_size = len(ret["results"]["bindings"])
            counter += 1
    except Exception as e:
        print("Error -->", e)
    return results


START_FROM_TEMPLATE = """SELECT DISTINCT ?uri WHERE { {0} ?x ?y .}"""


@DeprecationWarning
def get_all_triplets_from_start(start_node, paged=True):
    query_command = f"SELECT DISTINCT ?x ?y WHERE {{ <{start_node}> ?x ?y}} ORDER BY DESC(?x) DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def get_all_triplets_from_start2(start_node, relation=None, paged=True):
    """
    Given a start node and optionally a relation lists all of the triples.
    Only works on a single node and a single relation.
    :param start_node:
    :param relation:
    :param paged:
    :return:
    """
    if relation is None:
        query_command = f"SELECT DISTINCT ?x ?y WHERE {{ <{start_node}> ?x ?y}} ORDER BY DESC(?x) DESC(?y)"
    else:
        query_command = f"SELECT DISTINCT ?x ?y WHERE {{ <{start_node}> <{relation}> ?y}} ORDER BY DESC(?x) DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def get_all_triplets_to_start(relation, end_node, paged=True):
    """
    Given a relation and an end node returns all the triples
    :param relation:
    :param end_node:
    :param paged:
    :return:
    """
    query_command = f"SELECT DISTINCT ?x  WHERE {{ ?x <{relation}> <{end_node}>}} ORDER BY DESC(?x)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def count_all_triplets_to_start(relation, end_node):
    """
    given a relation and an end node counts how many triples there are.
    :param relation:
    :param end_node:
    :return:
    """
    query_command = f"SELECT COUNT(?x)  WHERE {{ ?x <{relation}> <{end_node}>}}"
    return run_sparql_query(query_command)


def do_excludes(excludes):
    if len(excludes) == 0:
        return ""
    # f"FILTER NOT EXISTS {{ ?y {ex[0]} {ex[1]} . }}"
    # f"FILTER NOT EXISTS {{ {' '.join(['?y <'+ ex[0] + '> <' + ex[1] +'> .' for ex in excludes])} }}"
    return f"FILTER NOT EXISTS {{ {' '.join(['?y <{}> <{}> . '.format(ex[0], ex[1]) if len(ex) == 2 else '{} <{}> <{}> . '.format(ex[0], ex[1], ex[2]) for ex in excludes])} }}"


def reltype_filter(reltype_excludes):
    if len(reltype_excludes) == 0:
        return ""
    return "FILTER( " + " || ".join(["?x != {}".format(excl) for excl in reltype_excludes]) + " )"


def get_all_triplets_from_w_excludes(start_node, excludes=[], reltype_excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?x ?y WHERE {{ <{start_node}> ?x ?y . {do_excludes(excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x) DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def add_answer_to_questions(input_df):
    questions = list()
    for i in range(len(input_df)):
        print(i)
        result = run_sparql_query(input_df[i]["sparql_query"])

        result_extracted = list()
        if result is not None:
            for item in result:
                if "callret-0" in item:
                    result_extracted.append(item["callret-0"]["value"])
                elif "uri" in item:
                    result_extracted.append(item["uri"]["value"])
                else:
                    print("New enemy encountered: ", item)
        x = input_df[i]
        x["result"] = result_extracted
        questions.append(x)

    return questions


def do_excludes2(excludes):
    if len(excludes) == 0:
        return ""
    # f"FILTER NOT EXISTS {{ ?y {ex[0]} {ex[1]} . }}"
    # f"FILTER NOT EXISTS {{ {' '.join(['?y <'+ ex[0] + '> <' + ex[1] +'> .' for ex in excludes])} }}"
    return f"FILTER NOT EXISTS {{ {' '.join([construct_exclusion(ex) for ex in excludes])} }}"


def construct_exclusion(elements):
    if elements[0] == "" and elements[1] == "":
        return '?y ?z <{}> . '.format(elements[2])
    elif elements[1] == "" and elements[2] == "":
        return '<{}> ?z ?y . '.format(elements[0])
    elif elements[0] == "":
        return '?y <{}> <{}> . '.format(elements[1], elements[2])
    elif elements[2] == "":
        return '<{}> <{}> ?y . '.format(elements[0], elements[1])


# TODO clean all of this py file!
def after_path_exclude(results, exclude_paths):

    good_elements = list()

    for res in results:
        good_flag = True
        for excl in exclude_paths:
            if res["y"]["value"] == excl[2]:
                good_flag = False
        if good_flag:
            good_elements.append(res)

    return good_elements


def get_all_triplets_from_s_excludes(start_node, node_excludes=[], reltype_excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?x ?y WHERE {{ <{start_node}> ?x ?y . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x) DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def get_all_triplets_from_s_excludes_ronly(start_node, node_excludes=[], reltype_excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?x WHERE {{ <{start_node}> ?x ?y . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def get_all_triplets_from_sr_excludes(start_node, relation, excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?y WHERE {{ <{start_node}> <{relation}> ?y . {do_excludes2(excludes)} }}  ORDER BY DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def get_all_triplets_from_o_excludes(end_node, node_excludes=[], reltype_excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?x ?y WHERE {{ ?x ?y <{end_node}>  . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?x) DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def get_all_triplets_from_o_excludes_ronly(end_node, node_excludes=[], reltype_excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?y WHERE {{ ?x ?y <{end_node}>  . {do_excludes2(node_excludes)} {reltype_filter(reltype_excludes)} }}  ORDER BY DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)


def get_all_triplets_from_ro_excludes(end_node, relation, excludes=[], paged=True):
    query_command = f"SELECT DISTINCT ?y WHERE {{ ?y <{relation}> <{end_node}> . {do_excludes2(excludes)} }}  ORDER BY DESC(?y)"
    if paged:
        return run_sparql_query_paged(query_command)
    else:
        return run_sparql_query(query_command)
