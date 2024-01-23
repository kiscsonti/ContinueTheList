from algorithm.bert_vectors import get_model, transform_name
import pickle
import random
import math
from tqdm import tqdm

with open("/home/kardosp/continuethelist/stats/extra_data/dbpedia_entities.pickle", "rb") as f:
    all_nodes = pickle.load(f)

filtered_all_nodes = list(filter(lambda x: "Category" not in x and len(x) <= 122, all_nodes))

print(len(all_nodes), len(filtered_all_nodes))

bert_model = get_model()

batch_size = 256
node2vector = dict()

for i in tqdm(range(math.ceil(len(filtered_all_nodes)/batch_size))):
    # print(i)
    subset = filtered_all_nodes[i*batch_size:(i+1)*batch_size]
    subset_transformed = [transform_name(item) for item in subset]
    embeddings = bert_model.encode(subset_transformed)
    # print("Embedding done!")

    for j in range(len(subset)):
        node2vector[subset[j]] = embeddings[j].tolist()

with open("/home/kardosp/continuethelist/stats/extra_data/dbpedia_vectors_MiniLM.pickle", "wb") as f:
    pickle.dump(node2vector, f)

output_path = "/home/kardosp/continuethelist/stats/extra_data/dbpedia_gensimvectors_MiniLM.kv"

with open(output_path, "w") as f:
    f.write(f"{len(node2vector)} {len(node2vector[list(node2vector.keys())[0]])}\n")

    for k, v in node2vector.items():
        f.write(f"{k} {' '.join([str(item) for item in v])}\n")
