import random
from sklearn.model_selection import train_test_split
from copy import copy


def split_on_seed_dataset(dataset, sample_size=4, seed=42, colname="result_urlonly"):
    dataset2 = copy(dataset)
    random.seed = seed
    for item in dataset2:
        if len(item[colname]) <= sample_size:
            item["seed"] = item[colname]
            item["gold"] = []
        else:
            record_seed, record_gold = train_test_split(item[colname], train_size=sample_size, random_state=seed)
            item["seed"] = record_seed
            item["gold"] = record_gold
    return dataset2


def split_on_record_dataset(dataset, seed=42):
    train_ds, test_ds = train_test_split(dataset, test_size=0.2, random_state=seed)
    return train_ds, test_ds


def split_dataset(dataset, sample_size=4, seed=42, colname="result_urlonly"):

    random.seed = seed
    for item in dataset:
        if len(item[colname]) <= sample_size:
            item["seed"] = item[colname]
            item["gold"] = []
        else:
            record_seed, record_gold = train_test_split(item[colname], train_size=sample_size, random_state=seed)
            item["seed"] = record_seed
            item["gold"] = record_gold
    train_ds, test_ds = train_test_split(dataset, test_size=0.2, random_state=seed)

    return train_ds, test_ds