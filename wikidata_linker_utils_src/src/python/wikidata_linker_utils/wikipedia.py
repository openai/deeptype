import re

import numpy as np

from os.path import join
from epub_conversion import convert_wiki_to_lines
from epub_conversion.wiki_decoder import almost_smart_open
from .wikipedia_language_codes import LANGUAGE_CODES
from .file import true_exists
from .bash import execute_bash
from .successor_mask import (
    load_redirections, match_wikipedia_to_wikidata
)


BADS = ["Wikipedia:", "Wikipédia:", "File:", "Media:", "Help:", "User:"]


def _lines_extractor(lines, article_name):
    """
    Simply outputs lines
    """
    yield (article_name, lines)


def _bad_link(link):
    return any(link.startswith(el) for el in BADS)


def iterate_articles(path):
    num_articles = 9999999999999
    with almost_smart_open(path, "rb") as wiki:
        for article_name, lines in convert_wiki_to_lines(
                wiki,
                max_articles=num_articles,
                clear_output=True,
                report_every=100,
                parse_special_pages=True,
                skip_templated_lines=False,
                line_converter=_lines_extractor):
            if not _bad_link(article_name):
                yield (article_name, lines)


def induce_wikipedia_prefix(wikiname):
    if wikiname in {code + "wiki" for code in LANGUAGE_CODES}:
        return wikiname
    else:
        raise ValueError("Could not determine prefix for wiki "
                         "with name %r." % (wikiname,))


def convert_sql_to_lookup(props, propname):
    propname = b",'" + propname.encode("utf-8") + b"','"
    ending = b"',"
    starting = b"("
    lookup = {}
    offset = 0
    while True:
        newpos = props.find(propname, offset)
        if newpos == -1:
            break
        begin = props.rfind(starting, offset, newpos)
        end = props.find(ending, newpos + len(propname))
        key = props[begin + len(starting):newpos]
        value = props[newpos + len(propname):end]
        lookup[key.decode('utf-8')] = value.decode('utf-8')
        offset = end
    return lookup


def load_wikipedia_pageid_to_wikidata(data_dir):
    fname = join(data_dir, "enwiki-latest-page_props.sql")
    if not true_exists(fname):
        execute_bash(
            "wget -O - https://dumps.wikimedia.org/enwiki/"
            "latest/enwiki-latest-page_props.sql.gz | gunzip > %s" % (fname,)
        )
    with open(fname, "rb") as fin:
        props = fin.read()
    return convert_sql_to_lookup(props, "wikibase_item")


link_pattern = re.compile(r'\[\[([^\]\[:]*)\]\]')


class WikipediaDoc(object):
    def __init__(self, doc):
        self.doc = doc

    def links(self, wiki_trie, redirections, prefix):
        current_pos = 0
        for match in re.finditer(link_pattern, self.doc):
            match_string = match.group(1)
            start = match.start()
            end = match.end()
            if current_pos != start:
                yield self.doc[current_pos:start], None
            current_pos = end

            if "|" in match_string:
                link, anchor = match_string.rsplit("|", 1)
                link = link.strip().split("#")[0]
            else:
                anchor = match_string
                link = anchor.strip()

            if len(link) > 0:
                dest_index = match_wikipedia_to_wikidata(
                    link,
                    wiki_trie,
                    redirections,
                    prefix
                )
                yield anchor, dest_index
            else:
                yield anchor, None
        if current_pos != len(self.doc):
            yield self.doc[current_pos:], None


def load_wikipedia_docs(path, size):
    docs = []
    for article_name, doc in iterate_articles(path):
        docs.append(WikipediaDoc(doc))
        if len(docs) == size:
            break
    return docs


def transition_trie_index(anchor_idx, dest_index, transitions, all_options):
    """
    Recover the new trie index for an index that has gone stale.
    Use a transitions array to know how original anchors now map to
    new trie indices.
    """
    option_transitions = transitions[anchor_idx]
    dest_index = option_transitions[option_transitions[:, 0] == dest_index, 1]
    if len(dest_index) == 0:
        dest_index = -1
    else:
        dest_index = np.asscalar(dest_index)
    if dest_index != -1:
        if not np.any(all_options == dest_index):
            dest_index = -1
    return dest_index


__all__ = ["load_redirections", "induce_wikipedia_prefix",
           "load_wikipedia_docs", "WikipediaDoc",
           "transition_trie_index", "iterate_articles"]
