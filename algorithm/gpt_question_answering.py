from fuzzywuzzy import fuzz
from loaders.loaders import *
from evaluation.split_dataset import split_on_seed_dataset
import json
from openai import OpenAI
from tqdm import tqdm
import os
import numpy as np


def get_tp(gold, pred, threshold=80):
    tp = list()
    gold = set(gold)
    for item in pred:
        for g in gold:
            score = fuzz.ratio(item, g)
            if score >= threshold:
                tp.append(item)
                break
    return tp


def get_fn(gold, pred, threshold=80):

    fn = list()
    pred = set(pred)
    for item in gold:
        best_score = 0
        for p in pred:
            score = fuzz.ratio(item, p)
            best_score = max(score, best_score)
        if best_score < threshold:
            fn.append(item)
    return fn


def get_fp(gold, pred, threshold=80):
    fp = list()
    gold = set(gold)
    for item in pred:
        best_score = 0
        for g in gold:
            score = fuzz.ratio(item, g)
            best_score = max(score, best_score)
        if best_score < threshold:
            fp.append(item)
    return fp


def get_prf(gold, prediction):
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


def flatten_json(json_in):
    json_out = list()
    if isinstance(json_in, list):
        return json_in
    if isinstance(json_in, dict):
        for k, v in json_in.items():
            json_out.extend(v)
    return json_out


def query_gpt_api(client, text):

    completion = client.chat.completions.create(
        # model="gpt-4o-mini",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": text}
        ],
        max_tokens=4096,
    )
    return completion.choices[0].message.content


def construct_prompt(question, seed):
    seed_str = ", ".join(seed)
    return f"""TASK: List all the entities that fulfill the stated question as if the year was 2016! To help you in the task we provide 4 example entities that are correct answers from DBPedia. We can also guarantee that the number of correct entities are between 8 and 100 and all of them are present in DBPedia 2016 version. Please answer in a JSON list format with all the entities that are correct answers for the question!
QUESTION: {question}
EXAMPLES: {seed_str}
JSON:"""


def clean_element_name(element):
    return element.split("/")[-1]


def response2json_flatten(response):
    if "{" in response and "}" in response:
        json_data = json.loads(response[response.find("{"):response.rfind("}")+1])
        result = list()
        if isinstance(json_data, dict):
            for k, v in json_data.items():
                result.extend(v)
        elif isinstance(json_data, list):
            result.extend(list(json_data))
        return json_data

    elif "[" in response and "]" in response:
        return json.loads(response[response.find("["):response.rfind("]")+1])

api_key = ""
client = OpenAI(api_key=api_key)
SAMPLE_SIZE = 4
SEED = 40

lc_quad_all = load_on_path(ds_lc_quad_unioned_cleaned)
splitted_lcquad = split_on_seed_dataset(lc_quad_all, SAMPLE_SIZE, SEED)

container = dict()

output_path = "/home/kardosp/continuethelist/gpt/question_answer/question_answer_v2.json"

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        container = json.load(f)


for row in (pbar := tqdm(splitted_lcquad)):
    if row["_id"] in container:
        continue
    question = row["corrected_question"]
    seed = row["seed"]
    cleaned_gold = list(map(clean_element_name, row["gold"]))
    in_prompt = construct_prompt(question, list(map(clean_element_name, seed)))
    response = query_gpt_api(client, in_prompt)
    try:
        response1_json = response2json_flatten(response)
        response1_flattened_preprocessed = list(map(clean_element_name, response1_json))
        p, r, f = get_prf(cleaned_gold, response1_flattened_preprocessed)
    except Exception as e:
        p = 0
        r = 0
        f = 0
    container[row["_id"]] = [response, [p, r, f]]
    pbar.set_description(f"F1 {str(f)}")

    with open(output_path, "w") as f:
        json.dump(container, f)


print("Precision:", np.mean([elem[1][0] for elem in container.values()]))
print("Recall:", np.mean([elem[1][1] for elem in container.values()]))
print("F1:", np.mean([elem[1][2] for elem in container.values()]))
