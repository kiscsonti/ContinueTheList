import os
import sys
from fuzzywuzzy import fuzz
from loaders.loaders import *
from openai import OpenAI
from evaluation.split_dataset import split_on_seed_dataset
import json
from tqdm import tqdm
import numpy as np


def parse_json(response: str):
    start_index = response.find("[")
    end_index = response.rfind("]")
    return response[start_index:end_index + 1]


def reformat_golds(gold):
    reformatted = list()
    for triple in gold:
        new_triple = list()
        for element in triple:
            new_triple.append(element.replace("<", "").replace(">", ""))
        reformatted.append(new_triple)
    return reformatted


def reformat_json(response):
    reformatted = list()
    for triple in response:
        new_triple = [triple["subject"], triple["predicate"], triple["object"]]
        reformatted.append(new_triple)
    return reformatted


def equals(a, b, threshold1=90, threshold2=80):
    if a.startswith("?"):
        a = "variable"
    if b.startswith("?"):
        b = "variable"

    a_cleaned = a[a.rfind("/") + 1:]
    b_cleaned = b[b.rfind("/") + 1:]

    score1 = fuzz.ratio(a, b)
    score2 = fuzz.ratio(a_cleaned, b_cleaned)

    # print(a, b, score1)
    # print(a_cleaned, b_cleaned, score2)
    if score2 >= threshold2 or score1 >= threshold1:
        return True
    return False


def compare_response_gold(response_json, gold):
    reformat_gold = reformat_golds(gold)
    correct_json = reformat_json(response_json)
    found_triples = list()
    for gold_triple in reformat_gold:
        for response_triple in correct_json:
            flag = True
            for i in range(3):
                if not equals(response_triple[i], gold_triple[i]):
                    flag = False
                    break
            if flag:
                found_triples.append(gold_triple)
                break
    return found_triples


def query_gpt_api(client, text):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": text}
        ],
        max_tokens=4096,
    )
    return completion.choices[0].message.content


def format_prompt2(input_list):
    return prompt2 + "\n".join(input_list) + "\nJSON:"


SAMPLE_SIZE = 4
SEED = 40

lc_quad_all = load_on_path(ds_lc_quad_unioned_cleaned)
splitted_lcquad = split_on_seed_dataset(lc_quad_all, SAMPLE_SIZE, SEED)

api_key = ""

client = OpenAI(
    api_key=api_key)

prompt2 = """Given a seed set of 4 nodes from a DBPedia 2016 dump, list all the connections that connect the seed elements so that it can be used to list additional similar nodes. Even though you do not have access to the database try to answer with a formatted JSON listing the triplets and nothing else! We list some examples with only a few connections but list as many connections you can. Make the list atleast 20 long!
<EXAMPLE 1>
Seed elements:
'http://dbpedia.org/resource/Maximilian_Fretter-Pico',
'http://dbpedia.org/resource/Joachim_von_Ribbentrop',
'http://dbpedia.org/resource/Günther_von_Kluge',
'http://dbpedia.org/resource/Wolfgang_von_Kluge
JSON:
[
    {
        "subject": "?x",
        "predicate": "http://dbpedia.org/ontology/battle",
        "object": "http://dbpedia.org/resource/Operation_Barbarossa"
    },
    {
        "subject": "?x",
        "predicate": "http://dbpedia.org/ontology/relation",
        "object": "?seed"
    },
    {
        "subject": "?x",
        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "object": "http://dbpedia.org/ontology/Person"
    },
]
</EXAMPLE 1>
<EXAMPLE 2>
Seed elements:
'http://dbpedia.org/resource/Buick_Roadmaster',
'http://dbpedia.org/resource/Chevrolet_Impala',
'http://dbpedia.org/resource/Pontiac_G8',
'http://dbpedia.org/resource/Chevrolet_Lumina'
JSON:
[
    {
        "subject": "http://dbpedia.org/resource/Cadillac_Fleetwood",
        "predicate": "http://dbpedia.org/property/related",
        "object": "?x"
    },
    {
        "subject": "?x",
        "predicate": "http://dbpedia.org/ontology/predecessor",
        "object": "?seed"
    },
    {
        "subject": "?x",
        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "object": "http://dbpedia.org/ontology/Automobile"
    },
]
</EXAMPLE 2>
Seed elements:
 """

output_path = "/home/kardosp/continuethelist/gpt/triplet_gen/triplet_gen_v3.json"

container = dict()

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        container = json.load(f)

for row in (pbar := tqdm(splitted_lcquad)):
    if row["_id"] in container:
        continue
    response = query_gpt_api(client, format_prompt2(row["seed"]))
    try:
        response_json = parse_json(response)
        res = compare_response_gold(json.loads(response_json), row["graph"])
        triplet_overlap_percent = len(res) / len(row["graph"])
    except Exception as e:
        triplet_overlap_percent = 0
    container[row["_id"]] = [response, triplet_overlap_percent]
    pbar.set_description(f"Last overlap {str(triplet_overlap_percent)}")

    with open(output_path, "w") as f:
        json.dump(container, f)


print("Average:", np.mean([elem[1] for elem in container.values()]))
