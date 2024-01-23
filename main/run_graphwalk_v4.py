from algorithm.graphwalk_functions_v4 import (run_on_dataset,
                                              run_grapwalk_function_v4)
from excludes.basic_excludes import (node_excl_yago_func, node_excl_extra,
                                     node_excl_owlthing_func, node_excl_wiki_func, rel_excl_wiki_func,
                                     node_english_only_func, long_node_exclude_func)
from loaders.loaders import *
from evaluation.split_dataset import split_on_seed_dataset
import json
import pickle
from time import time
from datetime import timedelta

start_time = time()

exclude_paths = []
relation_excludes = ["http://xmlns.com/foaf/0.1/primaryTopic",
                     'http://xmlns.com/foaf/0.1/isPrimaryTopicOf',
                     "http://dbpedia.org/property/isCitedBy",
                     "http://dbpedia.org/ontology/abstract",
                     "http://www.w3.org/2000/01/rdf-schema#comment",
                     'http://www.w3.org/ns/prov#wasDerivedFrom',
                     'http://xmlns.com/foaf/0.1/Document',
                     'http://purl.org/dc/terms/subject',
                     'http://purl.org/dc/elements/1.1/language',
                     'http://www.w3.org/2002/07/owl#Class',
                     'http://www.w3.org/2004/02/skos/core#Concept',
                     'http://purl.org/linguistics/gold/hypernym',

                     'http://www.w3.org/2000/01/rdf-schema#label',
                     'http://www.w3.org/2002/07/owl#sameAs',
                     'http://xmlns.com/foaf/0.1/name',
                     'http://dbpedia.org/ontology/wikiPageExternalLink',
                     'http://dbpedia.org/property/name',
                     'http://xmlns.com/foaf/0.1/depiction',
                     'http://dbpedia.org/ontology/thumbnail',
                     'http://xmlns.com/foaf/0.1/homepage',
                     'http://www.w3.org/2003/01/geo/wgs84_pos#long',
                     'http://www.w3.org/2003/01/geo/wgs84_pos#lat',
                     'http://www.georss.org/georss/point',

                     "http://dbpedia.org/property/wordnet_type",
                     'http://dbpedia.org/ontology/utcOffset',
                     'http://dbpedia.org/property/subdivisionType',
                     'http://dbpedia.org/ontology/background',
                     'http://dbpedia.org/property/blankName',

                     ]


res_excludes_inside_point_relation = [
    "http://purl.org/linguistics/gold/hypernym",
]

forward_step1_rel_excludes = ["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
backward_step1_rel_excludes = ["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
res_excludes_inside_point_node = []
node_excludes_funcs = [node_excl_wiki_func, node_excl_owlthing_func, node_excl_yago_func, node_excl_extra,
                       node_english_only_func, long_node_exclude_func]
relation_excludes_funcs = [rel_excl_wiki_func]

SEED = 44
SAMPLE_SIZE = 4
verbose = 1

output = f"/home/kardosp/continuethelist/outputs/graphwalk_v4_output_sample{SAMPLE_SIZE}_v1.pickle"

lc_quad_train = load_on_path(ds_lc_quad_train_cleaned)
splitted_lcquad = split_on_seed_dataset(lc_quad_train, SAMPLE_SIZE, SEED)


for i, record in enumerate(splitted_lcquad):
    print(i)
    if "output" in record:
        continue
    result, explanation = run_grapwalk_function_v4(record["seed"], exclude_paths, relation_excludes,
                                                   node_excludes_funcs,
                                                   forward_step1_rel_excludes,
                                                   backward_step1_rel_excludes,
                                                   relation_excludes_funcs,
                                                   verbose=verbose)
    record["output"] = result
    record["explanation"] = explanation

    with open(output, "wb") as f:
        pickle.dump(splitted_lcquad, f)

end_time = time()
elapsed_time = end_time-start_time
with open(output.replace("outputs", "outputs_stats"), "w") as f:
    json.dump({"runtime": elapsed_time}, f)

print(timedelta(seconds=elapsed_time))
