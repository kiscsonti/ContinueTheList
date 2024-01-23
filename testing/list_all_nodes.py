import os
import sys
import pickle
from query.query_functions2 import run_sparql_query, run_sparql_query_paged


all_nodes_count = int(run_sparql_query("SELECT DISTINCT count(?node) WHERE { ?node <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?b }")[0]['callret-0']["value"])

all_nodes = list()
full_query = True
i = 0
while full_query:
    print(i, len(all_nodes))
    subset = run_sparql_query("SELECT DISTINCT ?node WHERE { ?node <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?b } LIMIT 10000 offset " + str(i*10000))
    subset = [item["node"]["value"] for item in subset]
    if len(subset) != 10000:
        full_query = False

    filtered = list(filter(lambda x: x.startswith("http://dbpedia.org/"), subset))
    all_nodes.extend(filtered)
    # for element in filtered:
    #     all_nodes.append(element)

    if i % 100 == 0:
        with open("/home/kardosp/continuethelist/stats/extra_data/dbpedia_entities2.pickle", "wb") as f:
            pickle.dump(all_nodes, f)

    i += 1

with open("/home/kardosp/continuethelist/stats/extra_data/dbpedia_entities2.pickle", "wb") as f:
    pickle.dump(all_nodes, f)
