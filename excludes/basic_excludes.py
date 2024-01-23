import re


def node_excl_wiki_func(node):
    last_part = node.split("/")[-1]
    # print("Last part:", node)
    # print("Last part:", last_part)
    if "wikidata" in last_part.lower():
        return True
    if last_part.startswith("Q") or last_part.startswith("m."):
        try:
            int(last_part[1:])
            return True
        except Exception as e:
            pass
    return False


def node_excl_yago_func(node):
    if "/yago/" in node.lower():
        return True
    return False


def node_english_only_func(node):
    if re.search("\w+.dbpedia.org", node) is not None:
        return True
    return False


def long_node_exclude_func(node):
    if len(node) >= 100:
        return True
    return False


extra_exact_excl_list = ["dbtax", 'http://dbpedia.org/dbtax/List', 'http://dbpedia.org/ontology/List', ]
extra_like_excl_list = ['http://www.ontologydesignpatterns.org/', 'http://commons.wikimedia.org/',
                        'http://umbel.org/umbel/', 'http://schema.org',
                        "yago-knowledge.org/", "http://dbpedia.org/dbtax/Redirect", "/geo/", "Category:",
                        "mappings.dbpedia.org", "purl.org/", ]


def node_excl_extra(node):
    if node in extra_exact_excl_list:
        return True
    for like in extra_like_excl_list:
        if like in node:
            return True
    return False


def node_excl_owlthing_func(node):
    if "owl#thing" in node.lower():
        return True
    return False


def rel_excl_wiki_func(node):
    if "wiki" in node.lower():
        return True
    return False

