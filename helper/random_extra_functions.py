import pandas as pd
from evaluation.evaluate_script import normalize_triple


def explanation_to_pd(explanation):
    pd_rows = list()

    for k, v in explanation.items():
        if k == "invalids":
            continue
        if k == "backward2":
            for item in v:
                for triple in item[0]:
                    tmp = normalize_triple(triple)
                    tmp.append(k)
                    pd_rows.append(tmp)
        else:
            for triple in v:
                tmp = normalize_triple(triple)
                tmp.append(k)
                pd_rows.append(tmp)
    # return pd_rows
    return pd.DataFrame(pd_rows, columns=["start", "relation", "end", "type"])


