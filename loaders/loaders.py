import json
import pickle
ds_lc_quad = "/home/kardosp/continuethelist/datasets/LC-QuAD/"
ds_lc_quad_train = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_dump_valid_train.json"
ds_lc_quad_test = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_dump_valid_test.json"
ds_lc_quad_train_wgraph = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_dump_valid_w_graph_train.json"
ds_lc_quad_test_wgraph = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_dump_valid_w_graph_test.json"
ds_lc_quad_train_cleaned = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_dump_valid_w_graph_cleaned_min8_train.json"
ds_lc_quad_test_cleaned = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_dump_valid_w_graph_cleaned_min8_test.json"
ds_lc_quad_unioned_cleaned = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_dump_valid_w_graph_cleaned_min8_all.json"
ds_qald = "/home/kardosp/continuethelist/datasets/QALD/"
ds_qald_train = "/home/kardosp/continuethelist/datasets/QALD/qald9_dump_w_answers_train.json"
ds_qald_test = "/home/kardosp/continuethelist/datasets/QALD/qald9_dump_w_answers_test.json"

ds_lc_quad_train_chatgpt = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_cleaned_train_gptsample.json"
ds_lc_quad_test_chatgpt = "/home/kardosp/continuethelist/datasets/LC-QuAD/lc_quad_cleaned_test_gptsample.json"


def load_on_path(path):
    with open(path, "r") as f:
        lcquad_train = json.load(f)
    return lcquad_train


def load_on_path_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lc_quad_train():
    with open(ds_lc_quad_train, "r") as f:
        lcquad_train = json.load(f)
    return lcquad_train


def load_lc_quad_test():
    with open(ds_lc_quad_test, "r") as f:
        lcquad_test = json.load(f)
    return lcquad_test


def load_qald_train():
    with open(ds_qald_train, "r") as f:
        qald_train = json.load(f)
    return qald_train


def load_qald_test():
    with open(ds_qald_test, "r") as f:
        qald_test = json.load(f)
    return qald_test
