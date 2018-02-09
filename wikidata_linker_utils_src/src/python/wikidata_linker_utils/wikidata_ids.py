from os.path import exists, join, dirname
import marisa_trie
import json
from .file import true_exists
from os import makedirs


class MarisaAsDict(object):
    def __init__(self, marisa):
        self.marisa = marisa

    def get(self, key, fallback):
        value = self.marisa.get(key, None)
        if value is None:
            return fallback
        else:
            return value[0][0]

    def __getitem__(self, key):
        value = self.marisa[key]
        return value[0][0]

    def __contains__(self, key):
        return key in self.marisa


def load_wikidata_ids(path, verbose=True):
    wikidata_ids_inverted_path = join(path, 'wikidata_ids_inverted.marisa')
    with open(join(path, "wikidata_ids.txt"), "rt") as fin:
        ids = fin.read().splitlines()
    if exists(wikidata_ids_inverted_path):
        if verbose:
            print("loading wikidata id -> index")
        name2index = MarisaAsDict(marisa_trie.RecordTrie('i').load(wikidata_ids_inverted_path))
        if verbose:
            print("done")
    else:
        if verbose:
            print("building trie")

        name2index = MarisaAsDict(
            marisa_trie.RecordTrie('i', [(name, (k,)) for k, name in enumerate(ids)])
        )
        name2index.marisa.save(wikidata_ids_inverted_path)
        if verbose:
            print("done")
    return (ids, name2index)


def load_names(path, num, prefix):
    names = {}
    errors = 0  # debug
    if num > 0:
        with open(path, "rt", encoding="UTF-8") as fin:
            for line in fin:
                try:
                    name, number = line.rstrip('\n').split('\t')
                except ValueError:
                    errors += 1
                number = int(number)
                if number >= num:
                    break
                else:
                    if name.startswith(prefix):
                        names[number] = name[7:]
        print(errors)  # debug
    return names


def sparql_query(query):
    import requests
    wikidata_url = "https://query.wikidata.org/sparql"
    response = requests.get(
        wikidata_url,
        params={
            "format": "json",
            "query": query
        }
    ).json()
    out = {}
    for el in response["results"]['bindings']:
        label = el['propertyLabel']['value']
        value = el['property']['value']
        if value.startswith("http://www.wikidata.org/entity/"):
            value = value[len("http://www.wikidata.org/entity/"):]
        out[value] = label
    return out


def saved_sparql_query(savename, query):
    directory = dirname(savename)
    makedirs(directory, exist_ok=True)
    if true_exists(savename):
        with open(savename, "rt") as fin:
            out = json.load(fin)
        return out
    else:
        out = sparql_query(query)
        with open(savename, "wt") as fout:
            json.dump(out, fout)
        return out


def property_names(prop_save_path):
    """"
    Retrieve the mapping between wikidata properties ids (e.g. "P531") and
    their human-readable names (e.g. "diplomatic mission sent").

    Returns:
        dict<str, str> : mapping from property id to property descriptor.
    """
    return saved_sparql_query(
        prop_save_path,
        """
        SELECT DISTINCT ?property ?propertyLabel
        WHERE
        {
            ?property a wikibase:Property .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        """
    )


def temporal_property_names(prop_save_path):
    """"
    Retrieve the mapping between wikidata properties ids (e.g. "P531") and
    their human-readable names (e.g. "diplomatic mission sent") only
    for fields that are time-based.

    Returns:
        dict<str, str> : mapping from property id to property descriptor.
    """
    return saved_sparql_query(
        prop_save_path,
        """
        SELECT DISTINCT ?property ?propertyLabel
        WHERE
        {
            ?property a wikibase:Property .
            {?property wdt:P31 wd:Q18636219} UNION {?property wdt:P31 wd:Q22661913} .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        """
    )
