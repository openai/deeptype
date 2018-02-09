import json
import warnings

from os.path import join, exists
from functools import lru_cache

import marisa_trie
import requests
import numpy as np

from .successor_mask import (
    successor_mask, invert_relation, offset_values_mask
)
from .offset_array import OffsetArray, SparseAttribute
from .wikidata_ids import (
    load_wikidata_ids, load_names, property_names, temporal_property_names
)
from . import wikidata_properties as wprop


class CachedRelation(object):
    def __init__(self, use, state):
        self.use = use
        self.state = state


@lru_cache(maxsize=None)
def get_name(wikidata_id):
    res = requests.get("https://www.wikidata.org/wiki/" + wikidata_id)
    el = res.text.find('<title>')
    el_end = res.text.find('</title>')
    return res.text[el + len('<title>'):el_end]


class TypeCollection(object):
    def __init__(self, path, num_names_to_load=100000, language_path=None, prefix="enwiki", verbose=True,
                 cache=True):
        self.cache = cache
        self.path = path
        self.verbose = verbose
        self.wikidata_names2prop_names = property_names(
            join(path, 'wikidata_property_names.json')
        )
        self.wikidata_names2temporal_prop_names = temporal_property_names(
            join(path, 'wikidata_time_property_names.json')
        )
        # add wikipedia english category links:
        self.wikidata_names2prop_names[wprop.CATEGORY_LINK] = "category_link"
        self.wikidata_names2prop_names[wprop.FIXED_POINTS] = "fixed_points"
        self.known_names = load_names(
            join(path, "wikidata_wikititle2wikidata.tsv"),
            num_names_to_load,
            prefix=prefix
        )
        self.num_names_to_load = num_names_to_load
        self.ids, self.name2index = load_wikidata_ids(path, verbose=self.verbose)
        self._relations = {}
        self._attributes = {}
        self._inverted_relations = {}
        self._article2id = None
        self._web_get_name = True
        self._satisfy_cache = {}

        # empty blacklist:
        self.set_bad_node(
            set(), set()
        )
        if language_path is not None:
            article_links = np.load(join(language_path, "trie_index2indices_values.npy"))
            article_links_counts = np.load(join(language_path, "trie_index2indices_counts.npy"))
            self._weighted_articles = np.bincount(article_links, weights=article_links_counts).astype(np.int32)
            if len(self._weighted_articles) != len(self.ids):
                self._weighted_articles = np.concatenate(
                    [
                        self._weighted_articles,
                        np.zeros(len(self.ids) - len(self._weighted_articles), dtype=np.int32)
                    ]
                )
        else:
            self._weighted_articles = None

    def attribute(self, name):
        if name not in self._attributes:
            is_temporal = name in self.wikidata_names2temporal_prop_names
            assert(is_temporal), "load relations using `relation` method."
            if self.verbose:
                print('load %r (%r)' % (name, self.wikidata_names2prop_names[name],))
            self._attributes[name] = SparseAttribute.load(
                join(self.path, "wikidata_%s" % (name,))
            )
        return self._attributes[name]

    @property
    def article2id(self):
        if self._article2id is None:
            if self.verbose:
                print('load %r' % ("article2id",))
            self._article2id = marisa_trie.RecordTrie('i').load(
                join(self.path, "wikititle2wikidata.marisa")
            )
            if self.verbose:
                print("done.")
        return self._article2id

    def relation(self, name):
        if name.endswith(".inv"):
            return self.get_inverted_relation(name[:-4])
        if name not in self._relations:
            is_temporal = name in self.wikidata_names2temporal_prop_names
            assert(not is_temporal), "load attributes using `attribute` method."
            if self.verbose:
                print('load %r (%r)' % (name, self.wikidata_names2prop_names[name],))
            self._relations[name] = OffsetArray.load(
                join(self.path, "wikidata_%s" % (name,)),
                compress=True
            )
        return self._relations[name]

    def set_bad_node(self, bad_node, bad_node_pair):
        changed = False
        if hasattr(self, "_bad_node") and self._bad_node != bad_node:
            changed = True
        if hasattr(self, "_bad_node_pair") and self._bad_node_pair != bad_node_pair:
            changed = True

        self._bad_node = bad_node
        self._bad_node_pair = bad_node_pair
        self._bad_node_array = np.array(list(bad_node), dtype=np.int32)

        bad_node_pair_right = {}
        for node_left, node_right in self._bad_node_pair:
            if node_right not in bad_node_pair_right:
                bad_node_pair_right[node_right] = [node_left]
            else:
                bad_node_pair_right[node_right].append(node_left)
        bad_node_pair_right = {
            node_right: np.array(node_lefts, dtype=np.int32)
            for node_right, node_lefts in bad_node_pair_right.items()
        }
        self._bad_node_pair_right = bad_node_pair_right

        if changed:
            self.reset_cache()

    def get_name(self, identifier):
        if identifier >= self.num_names_to_load and self._web_get_name:
            try:
                return get_name(self.ids[identifier]) + " (" + self.ids[identifier] + ")"
            except requests.exceptions.ConnectionError:
                self._web_get_name = False
        name = self.known_names.get(identifier, None)
        if name is None:
            return self.ids[identifier]
        else:
            return name + " (" + self.ids[identifier] + ")"

    def describe_connection(self, source, destination, allowed_edges):
        if isinstance(source, str):
            if source in self.name2index:
                source_index = self.name2index[source]
            else:
                source_index = self.article2id["enwiki/" + source][0][0]
        else:
            source_index = source

        if isinstance(destination, str):
            if destination in self.name2index:
                dest_index = self.name2index[destination]
            else:
                dest_index = self.article2id["enwiki/" + destination][0][0]
        else:
            dest_index = destination

        found_path = self.is_member_with_path(
            source_index,
            allowed_edges,
            [dest_index]
        )
        if found_path is not None:
            _, path = found_path
            for el in path:
                if isinstance(el, str):
                    print("    " + el)
                else:
                    print(self.get_name(el), el)
        else:
            print('%r and %r are not connected' % (source, destination))

    def is_member_with_path(self, root, fields, member_fields, max_steps=float("inf"), steps=0, visited=None, path=None):
        if steps >= max_steps:
            return None
        if visited is None:
            visited = set()

        if path is None:
            path = [root]
        else:
            path = path + [root]

        for field in fields:
            field_parents = self.relation(field)[root]
            for el in field_parents:
                if el in member_fields and el not in self._bad_node and (root, el) not in self._bad_node_pair:
                    return True, path + [field, el]
            for el in field_parents:
                if el in visited or el in self._bad_node or (root, el) in self._bad_node_pair:
                    continue
                visited.add(el)
                res = self.is_member_with_path(el, fields, member_fields, max_steps, steps=steps + 1, visited=visited, path=path + [field])
                if res is not None:
                    return res
        return None

    def get_inverted_relation(self, relation_name):
        if relation_name.endswith(".inv"):
            return self.relation(relation_name[:-4])
        if relation_name not in self._inverted_relations:
            new_values_path = join(self.path, "wikidata_inverted_%s_values.npy" % (relation_name,))
            new_offsets_path = join(self.path, "wikidata_inverted_%s_offsets.npy" % (relation_name,))

            if not exists(new_values_path):
                relation = self.relation(relation_name)
                if self.verbose:
                    print("inverting relation %r (%r)" % (relation_name, self.wikidata_names2prop_names[relation_name],))
                new_values, new_offsets = invert_relation(
                    relation.values,
                    relation.offsets
                )
                np.save(new_values_path, new_values)
                np.save(new_offsets_path, new_offsets)
            if self.verbose:
                print("load inverted %r (%r)" % (relation_name, self.wikidata_names2prop_names[relation_name]))
            self._inverted_relations[relation_name] = OffsetArray.load(
                join(self.path, "wikidata_inverted_%s" % (relation_name,)),
                compress=True
            )
        return self._inverted_relations[relation_name]

    def successor_mask(self, relation, active_nodes):
        if isinstance(active_nodes, list):
            active_nodes = np.array(active_nodes, dtype=np.int32)
        if active_nodes.dtype != np.int32:
            active_nodes = active_nodes.astype(np.int32)
        return successor_mask(
            relation.values, relation.offsets, self._bad_node_pair_right, active_nodes
        )

    def remove_blacklist(self, state):
        state[self._bad_node_array] = False

    def satisfy(self, relation_names, active_nodes, max_steps=None):
        assert(len(relation_names) > 0), (
            "relation_names cannot be empty."
        )
        if self.cache and isinstance(active_nodes, (list, tuple)) and len(active_nodes) < 100:
            satisfy_key = (tuple(sorted(relation_names)), tuple(sorted(active_nodes)), max_steps)
            if satisfy_key in self._satisfy_cache:
                cached = self._satisfy_cache[satisfy_key]
                cached.use += 1
                return cached.state
        else:
            satisfy_key = None
        inverted_relations = [self.get_inverted_relation(relation_name) for relation_name in relation_names]
        state = np.zeros(inverted_relations[0].size(), dtype=np.bool)
        state[active_nodes] = True
        step = 0
        while len(active_nodes) > 0:
            succ = None
            for relation in inverted_relations:
                if succ is None:
                    succ = self.successor_mask(relation, active_nodes)
                else:
                    succ = succ | self.successor_mask(relation, active_nodes)
            new_state = state | succ
            self.remove_blacklist(new_state)
            (active_nodes,) = np.where(state != new_state)
            active_nodes = active_nodes.astype(np.int32)
            state = new_state
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if satisfy_key is not None:
            self._satisfy_cache[satisfy_key] = CachedRelation(1, state)
        return state

    def reset_cache(self):
        cache_keys = list(self._satisfy_cache.keys())
        for key in cache_keys:
            if self._satisfy_cache[key].use == 0:
                del self._satisfy_cache[key]
            else:
                self._satisfy_cache[key].use = 0

    def print_top_class_members(self, truth_table, name="Other", topn=20):
        if self._weighted_articles is not None:
            print("%s category, highly linked articles in wikipedia:" % (name,))
            sort_weight = self._weighted_articles * truth_table
            linked_articles = int((sort_weight > 0).sum())
            print("%s category, %d articles linked in wikipedia:" % (name, linked_articles))
            top_articles = np.argsort(sort_weight)[::-1]
            for art in top_articles[:topn]:
                if not truth_table[art]:
                    break
                print("%r (%d)" % (self.get_name(art), self._weighted_articles[art]))
            print("")
        else:
            print("%s category, sample of members:" % (name,))
            top_articles = np.where(truth_table)[0]
            for art in top_articles[:topn]:
                print("%r" % (self.get_name(art),))
            print("")

    def class_report(self, relation_names, truth_table, name="Other", topn=20):
        active_nodes = np.where(truth_table)[0].astype(np.int32)
        num_active_nodes = len(active_nodes)
        print("%s category contains %d unique items." % (name, num_active_nodes,))
        relations = [self.relation(relation_name) for relation_name in relation_names]
        for relation, relation_name in zip(relations, relation_names):
            mask = offset_values_mask(relation.values, relation.offsets, active_nodes)
            counts = np.bincount(relation.values[mask])
            topfields = np.argsort(counts)[::-1]
            print("%s category, most common %r:" % (name, relation_name,))
            for field in topfields[:topn]:
                if counts[field] == 0:
                    break
                print("%.3f%% (%d): %r" % (100.0 * counts[field] / num_active_nodes,
                                           counts[field],
                                           self.get_name(field)))
            print("")

        is_fp = np.logical_and(
            np.logical_or(
                self.relation(wprop.FIXED_POINTS + ".inv").edges() > 0,
                self.relation(wprop.FIXED_POINTS).edges() > 0
            ),
            truth_table
        )
        self.print_top_class_members(
            is_fp, topn=topn, name=name + " (fixed points)"
        )
        if self._weighted_articles is not None:
            self.print_top_class_members(truth_table, topn=topn, name=name)

    def load_blacklist(self, path):
        with open(path, "rt") as fin:
            blacklist = json.load(fin)
        filtered_bad_node = []
        for el in blacklist["bad_node"]:
            if el not in self.name2index:
                warnings.warn("Node %r under `bad_node` is not a known wikidata id." % (
                    el
                ))
                continue
            filtered_bad_node.append(el)
        bad_node = set(self.name2index[el] for el in filtered_bad_node)

        filtered_bad_node_pair = []

        for el, oel in blacklist["bad_node_pair"]:
            if el not in self.name2index:
                warnings.warn("Node %r under `bad_node_pair` is not a known wikidata id." % (
                    el
                ))
                continue
            if oel not in self.name2index:
                warnings.warn("Node %r under `bad_node_pair` is not a known wikidata id." % (
                    oel
                ))
                continue
            filtered_bad_node_pair.append((el, oel))
        bad_node_pair = set([(self.name2index[el], self.name2index[oel])
                             for el, oel in filtered_bad_node_pair])
        self.set_bad_node(bad_node, bad_node_pair)
