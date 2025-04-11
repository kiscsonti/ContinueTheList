import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
# from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from gensim.models import KeyedVectors
from copy import copy
from scipy.spatial.distance import cdist


class Network(nn.Module):
    def __init__(self, emb_dim=384):
        super().__init__()
        self.emb_dim = emb_dim
        self.W = torch.nn.Parameter(torch.ones(self.emb_dim))
        self.W.requires_grad = True

    def forward(self, x):
        x = self.W * x
        return x


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = euclidean_distance(anchor, positive)
        distance_negative = euclidean_distance(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def euclidean_distance(x, y):
    """
    Compute Euclidean distance between two tensors.
    """
    return torch.pow(x - y, 2).sum(dim=1)


def compute_distance_matrix(anchor, positive, negative):
    """
    Compute distance matrix between anchor, positive, and negative samples.
    """
    distance_matrix = torch.zeros(anchor.size(0), 3)
    distance_matrix[:, 0] = euclidean_distance(anchor, anchor)
    distance_matrix[:, 1] = euclidean_distance(anchor, positive)
    distance_matrix[:, 2] = euclidean_distance(anchor, negative)
    return distance_matrix


def batch_hard_triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Compute triplet loss using the batch hard strategy.
    """
    distance_matrix = compute_distance_matrix(anchor, positive, negative)
    hard_negative = torch.argmax(distance_matrix[:, 2])
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin)
    loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0][hard_negative] - distance_matrix[:, 2] + margin)
    return torch.mean(loss)


def batch_all_triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Compute triplet loss using the batch all strategy.
    """
    distance_matrix = compute_distance_matrix(anchor, positive, negative)
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin)
    loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 2] + margin)
    return torch.mean(loss)


def get_center_vector(seed, wv_model):
    vectors = dict()
    missing = 0
    for item in seed:
        try:
            vectors[item] = list(wv_model[item])
        except Exception as e:
            missing += 1
    if missing == len(seed):
        return None, None, missing
    center_vector = np.mean(list(vectors.values()), axis=0)
    return center_vector, vectors, missing


class CoreDataset(Dataset):
    def __init__(self, anchor, positives, negatives, w, sample="all"):
        self.neg = negatives
        self.anchor = anchor
        self.pos = positives
        self.w = w
        if sample == "all":
            self.records = self.setup_triples_all()
        elif sample == "closest":
            self.records = self.setup_triples_closest()
        elif sample == "hardest":
            self.records = self.setup_triples_hardest()
        else:
            print("No sample like:", sample)
            self.records = self.setup_triples_all()

    def setup_triples_closest(self):
        triples = list()
        for neg_element in self.neg:
            closest_pos = None
            closest_dist = None
            for pos in self.pos:

                dist = weighted_euclidean(neg_element, pos, self.w)
                if closest_dist is None:
                    closest_dist = dist
                    closest_pos = pos
                else:
                    if closest_dist > dist:
                        closest_dist = dist
                        closest_pos = pos
            triples.append([self.anchor, closest_pos, neg_element])

        return triples

    def setup_triples_all(self):
        triples = list()
        for neg_element in self.neg:
            for pos in self.pos:
                triples.append([self.anchor, pos, neg_element])
        return triples

    def setup_triples_hardest(self):
        triples = list()
        for neg_element in self.neg:
            farthest_pos = None
            farthest_dist = None
            for pos in self.pos:

                dist = weighted_euclidean(self.anchor, pos, self.w)
                if farthest_dist is None:
                    farthest_dist = dist
                    farthest_pos = pos
                else:
                    if farthest_dist < dist:
                        farthest_dist = dist
                        farthest_pos = pos
            triples.append([self.anchor, farthest_pos, neg_element])
        return triples

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        return (torch.from_numpy(self.records[item][0]),
                torch.from_numpy(self.records[item][1]),
                torch.from_numpy(self.records[item][2]))

class WeightClipper(object):

    def __init__(self):
        pass

    def __call__(self, module):
        # filter the variables to get the ones you want
        print("Clipping")
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0.01, 1)
            module.weight.data = w


def generate_ds():
    pass


def train_net(model, train_set, ):
    pass


def weighted_euclidean(a, b, w):
    q = a-b
    return np.sqrt((w*q*q).sum())


def sample_closest_n(center, seed, w, n, wv_model):
    closest_elements = list()

    all_euclidean_dist = cdist([center], wv_model.vectors,
                               metric=lambda a, b: weighted_euclidean(a, b, w.clone().detach().numpy()))
    argsort_res = all_euclidean_dist.argsort()[0]
    for index in argsort_res:

        if len(closest_elements) == n:
            break

        name = wv_model.index_to_key[index]
        if name not in seed:
            closest_elements.append((wv_model.index_to_key[index], all_euclidean_dist[0][index]))
    return closest_elements


def annotate_elements(selection, gold):
    pos = list()
    neg = list()
    for element in selection:
        if element in gold:
            pos.append(element)
        else:
            neg.append(element)
    return pos, neg


def get_vectors(keys, wv_model):
    return np.array([wv_model[key] for key in keys])


def get_vectors2(keys, wv_model):
    vectors = list()
    missing = 0
    for item in keys:
        try:
            vec = list(wv_model[item])
            vectors.append(vec)
        except Exception as e:
            missing += 1
    if missing == len(keys):
        return None, missing
    return np.array(vectors)


def run_train_loop(dataloader, neuralnet, optimizer, criterion, epoch=1):
    for i in range(epoch):
        loss_sum = 0
        for step, (anchor_img, positive_img, negative_img) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            anchor_img = anchor_img.to("cpu")
            positive_img = positive_img.to("cpu")
            negative_img = negative_img.to("cpu")

            optimizer.zero_grad()
            anchor_out = neuralnet(anchor_img)
            positive_out = neuralnet(positive_img)
            negative_out = neuralnet(negative_img)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            #Clipping (0.001,1)
            for param in neuralnet.parameters():
                param.data = param.data.clip(0.001, 1)

    return loss_sum


def run_iterative(seed, wv_model: KeyedVectors, gold, annotation=5, iteration=3, epoch=1, gold_size=None,
                  algo_name="all"):
    """
    Input is a single record's seed.
    distance can be the following: ["circle", "euclidean", "ellipse", "l1ellipse"]
    :return:
    """
    
    center_vector, vectors, missing = get_center_vector(seed, wv_model)
    closer_elements = list()
    weight_net = Network(center_vector.shape[0])
    # optimizer = optim.Adam(weight_net.parameters(), lr=0.001)
    criterion = torch.jit.script(TripletLoss())
    #criterion = torch.nn.TripletMarginLoss(margin=1, p=2, eps=1e-7)
    optimizer = optim.Adam(weight_net.parameters(), lr=0.75)

    new_seed = copy(seed)

    #Repeat start
    for it in range(iteration):
        closest = sample_closest_n(center_vector, new_seed, weight_net.W, annotation, wv_model)
        closest_elements = [item[0] for item in closest]
        new_seed.extend(closest_elements)
        pos, neg = annotate_elements(new_seed, gold)
        pos_vectors = get_vectors2(pos, wv_model)
        neg_vectors = get_vectors2(neg, wv_model)
        train_ds = CoreDataset(center_vector, pos_vectors, neg_vectors, weight_net.W.clone().detach().numpy(),
                               sample=algo_name)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)

        loss = run_train_loop(train_loader, weight_net, optimizer, criterion, epoch)

        print("Iteration: {}/{} - Loss: {:.4f}".format(it+1, iteration, np.mean(loss)))

    pos, neg = annotate_elements(new_seed, gold)
    pos_vectors = get_vectors2(pos, wv_model)
    point_dist = cdist([center_vector], pos_vectors,
                       metric=lambda a, b: weighted_euclidean(a, b, weight_net.W.clone().detach().numpy()))
    threshold = np.max(point_dist)
    all_euclidean_dist = cdist([center_vector], wv_model.vectors,
                               metric=lambda a, b: weighted_euclidean(a, b, weight_net.W.clone().detach().numpy()))
    argsort_res = all_euclidean_dist.argsort()[0]

    for index in argsort_res:
        if all_euclidean_dist[0][index] > threshold:
            break
        if gold_size is not None and len(closer_elements) >= gold_size:
            break
        closer_elements.append((index, all_euclidean_dist[0][index]))

    prediction = [wv_model.index_to_key[item[0]] for item in closer_elements]

    return prediction, missing
