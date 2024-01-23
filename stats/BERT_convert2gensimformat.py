import pickle

input_path = "/home/kardosp/continuethelist/stats/extra_data/dbpedia_vectors_MiniLM.pickle"
output_path = "/home/kardosp/continuethelist/stats/extra_data/dbpedia_gensimvectors_MiniLM.kv"

with open(input_path, "rb") as f:
    node2vector = pickle.load(f)

with open(output_path, "w") as f:
    f.write(f"{len(node2vector)} {len(node2vector[list(node2vector.keys())[0]])}\n")
    
    for k, v in node2vector.items():
        f.write(f"{k} {' '.join([str(item) for item in v])}\n")
