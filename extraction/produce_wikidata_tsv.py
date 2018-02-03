"""
Create a tsv file where where the first column is a token and second column
is the QID (wikidata internal id for entities). This can then be used
by evaluate_learnability or from training a type model.

Usage
-----

```
python3 produce_wikidata_tsv.py configs/en_export_config.json en_wikipedia.tsv
```

Use `--relative_to` argument to specify the base directory for relative paths in the
config file.

"""
import argparse
import re
import json

from os.path import join, dirname, realpath, exists

import marisa_trie
import ciseau
import numpy as np

from wikidata_linker_utils.wikipedia import (
    iterate_articles, induce_wikipedia_prefix, load_redirections,
    transition_trie_index
)
from wikidata_linker_utils.json import load_config
from wikidata_linker_utils.offset_array import OffsetArray
from wikidata_linker_utils.type_collection import TypeCollection
from wikidata_linker_utils.anchor_filtering import acceptable_anchor, clean_up_trie_source
from wikidata_linker_utils.wikipedia import match_wikipedia_to_wikidata

SCRIPT_DIR = dirname(realpath(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("out")
    parser.add_argument("--relative_to", type=str, default=None)
    return parser.parse_args(args=args)


link_pattern = re.compile(r"\[\[([^\]\[:]*)\]\]")
ref_pattern = re.compile(r"<ref[^<>]*>[^<]+</ref>")
double_bracket_pattern = re.compile(r"{{[^{}]+}}")
title_pattern = re.compile(r"==+([^=]+)==+")
bullet_point_pattern = re.compile(r"^([*#])", re.MULTILINE)


def merge_tags(words, tags, start_sent):
    out = [(w, []) for w in words]
    for tag_start, tag_end, tag in tags:
        so_far = start_sent
        for k, word in enumerate(words):
            begins = tag_start <= so_far or (tag_start > so_far and tag_start < so_far + len(word))
            ends = (so_far + len(word) <= tag_end) or (tag_end < so_far + len(word) and tag_end > so_far)
            if begins and ends:
                out[k][1].append(tag)
            so_far += len(word)
            if so_far >= tag_end:
                break
    return out


def pick_relevant_tags(tagged_sequence, char_offset, char_offset_end):
    relevant_tags = []
    for word, tags in tagged_sequence:
        if tags is not None:
            start, end, dest_index = tags
            if start >= char_offset and start < char_offset_end:
                relevant_tags.append((start, end, dest_index))
            if start >= char_offset_end:
                break
    return relevant_tags


def convert_document_to_labeled_tags(annotated, sentences):
    paragraphs = []
    paragraph = []
    char_offset = 0
    for sentence in sentences:
        sentence_length = sum(len(w) for w in sentence)
        sentence_tags = pick_relevant_tags(
            annotated,
            char_offset,
            char_offset + sentence_length
        )
        sentence_with_tags = merge_tags(
            sentence,
            sentence_tags,
            char_offset
        )
        sentence_with_tags = [
            (
                w,
                [tags[0]] if len(tags) > 0 else []
            ) for w, tags in sentence_with_tags
        ]
        if "\n" in sentence[-1]:
            paragraph.extend(sentence_with_tags)
            paragraphs.append(paragraph)
            paragraph = []
        else:
            paragraph.extend(sentence_with_tags)
        char_offset += sentence_length
    if len(paragraph) > 0:
        paragraphs.append(paragraph)
    return paragraphs


def annotate_document(doc,
                      collection,
                      wiki_trie,
                      anchor_trie,
                      trie_index2indices,
                      trie_index2indices_counts,
                      trie_index2indices_transitions,
                      redirections,
                      prefix):
    out = []
    current_position = 0
    current_position_no_brackets = 0
    for match in re.finditer(link_pattern, doc):
        start = match.start()
        end = match.end()

        if current_position != start:
            out.append(
                (doc[current_position:start], None)
            )
            current_position_no_brackets += start - current_position
        current_position = end

        match_string = match.group(1).strip()
        if "|" in match_string:
            link, anchor = match_string.rsplit("|", 1)
            link = link.strip().split("#")[0]
            anchor = anchor
            anchor_stripped = anchor.strip()
        else:
            anchor = match_string
            anchor_stripped = match_string.strip()
            link = anchor_stripped

        if len(anchor) > 0 and len(link) > 0:
            anchor = clean_up_trie_source(anchor, lowercase=False)
            lowercase_anchor = anchor.lower()
            if acceptable_anchor(lowercase_anchor, anchor_trie):
                anchor_idx = anchor_trie[lowercase_anchor]
                dest_index = match_wikipedia_to_wikidata(link, wiki_trie, redirections, prefix)
                if dest_index is not None:
                    all_options = trie_index2indices[anchor_idx]
                    if len(all_options) > 0:
                        if trie_index2indices_transitions is not None:
                            dest_index = transition_trie_index(
                                anchor_idx, dest_index,
                                trie_index2indices_transitions,
                                all_options
                            )
                        try:
                            new_dest_index = dest_index
                            keep = True

                            if keep:
                                out.append(
                                    (
                                        anchor,
                                        (
                                            current_position_no_brackets,
                                            current_position_no_brackets + len(anchor),
                                            collection.ids[new_dest_index]
                                        )
                                    )
                                )
                                current_position_no_brackets += len(anchor)
                                continue
                        except IndexError:
                            # missing element
                            pass
        current_position_no_brackets += len(anchor)
        out.append(
            (anchor, None)
        )

    if current_position != len(doc):
        out.append(
            (doc[current_position:len(doc)], None)
        )
    return out


def convert(article_name,
            doc,
            collection,
            wiki_trie,
            anchor_trie,
            trie_index2indices,
            trie_index2indices_counts,
            trie_index2indices_transitions,
            redirections,
            prefix):
    doc = doc.replace("\t", " ")
    # remove ref tags:
    doc = re.sub(ref_pattern, " ", doc)
    doc = re.sub(double_bracket_pattern, " ", doc)
    doc = re.sub(title_pattern, r"\n\n\1\. ", doc)
    doc = re.sub(bullet_point_pattern, r"\1 ", doc)

    article_index = match_wikipedia_to_wikidata(
        article_name, wiki_trie, redirections, prefix
    )
    # find location of tagged items in wikipedia:
    annotated = annotate_document(doc,
                                  collection,
                                  wiki_trie,
                                  anchor_trie,
                                  trie_index2indices,
                                  trie_index2indices_counts,
                                  trie_index2indices_transitions,
                                  redirections,
                                  prefix)
    text_without_brackets = "".join(text for text, _ in annotated)
    sentences = ciseau.sent_tokenize(
        text_without_brackets,
        normalize_ascii=False,
        keep_whitespace=True
    )
    return (
        convert_document_to_labeled_tags(
            annotated, sentences
        ),
        collection.ids[article_index] if article_index is not None else "other"
    )



def main():
    args = parse_args()
    config = load_config(
        args.config,
        ["wiki", "language_path", "wikidata", "redirections"],
        defaults={
            "num_names_to_load": 0,
            "prefix": None,
            "sample_size": 100
        },
        relative_to=args.relative_to
    )
    prefix = config.prefix or induce_wikipedia_prefix(config.wiki)

    collection = TypeCollection(
        config.wikidata,
        num_names_to_load=0
    )
    collection.load_blacklist(join(SCRIPT_DIR, "blacklist.json"))

    trie_index2indices = OffsetArray.load(
        join(config.language_path, "trie_index2indices"),
        compress=True
    )
    trie_index2indices_counts = OffsetArray(
        np.load(join(config.language_path, "trie_index2indices_counts.npy")),
        trie_index2indices.offsets
    )
    if exists(join(config.language_path, "trie_index2indices_transition_values.npy")):
        trie_index2indices_transitions = OffsetArray(
            np.load(join(config.language_path, "trie_index2indices_transition_values.npy")),
            np.load(join(config.language_path, "trie_index2indices_transition_offsets.npy")),
        )
    else:
        trie_index2indices_transitions = None

    anchor_trie = marisa_trie.Trie().load(join(config.language_path, "trie.marisa"))
    wiki_trie = marisa_trie.RecordTrie('i').load(
        join(config.wikidata, "wikititle2wikidata.marisa")
    )
    redirections = load_redirections(config.redirections)

    seen = 0
    with open(args.out, "wt") as fout:
        try:
            for article_name, article in iterate_articles(config.wiki):
                fixed_article, article_qid = convert(
                    article_name,
                    article,
                    collection=collection,
                    anchor_trie=anchor_trie,
                    wiki_trie=wiki_trie,
                    trie_index2indices=trie_index2indices,
                    trie_index2indices_counts=trie_index2indices_counts,
                    trie_index2indices_transitions=trie_index2indices_transitions,
                    redirections=redirections,
                    prefix=prefix)
                for paragraph in fixed_article:
                    for word, qids in paragraph:
                        if len(qids) > 0:
                            fout.write(word.rstrip() + "\t" + "\t".join(qids + [article_qid]) + "\n")
                        else:
                            fout.write(word.rstrip() + "\n")
                    fout.write("\n")
                seen += 1
                if seen >= config.sample_size:
                    break
        finally:
            fout.flush()
            fout.close()


if __name__ == "__main__":
    main()

