import argparse

from os.path import join
from os import makedirs

import marisa_trie
import numpy as np

from wikidata_linker_utils.bash import count_lines
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.wikipedia import match_wikipedia_to_wikidata, load_redirections
from wikidata_linker_utils.successor_mask import construct_mapping, construct_anchor_trie


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("wikipedia2wikidata_trie",
        help="Location of wikipedia -> wikidata mapping trie.")
    parser.add_argument("prefix", type=str,
        help="What language is being processed, e.g. enwiki, frwiki, etc.")
    parser.add_argument("anchor_tags", type=str,
        help="Location where anchor tags were saved (tsv).")
    parser.add_argument("redirections", type=str,
        help="Location where redirections were saved (tsv).")
    parser.add_argument("out", type=str,
        help="Directory to save trie/data in.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    makedirs(args.out, exist_ok=True)
    wikipedia2wikidata_trie = marisa_trie.RecordTrie('i').load(
        args.wikipedia2wikidata_trie
    )
    print('loaded trie')
    redirections = load_redirections(args.redirections)
    anchor_trie = construct_anchor_trie(
        anchor_tags=args.anchor_tags,
        wikipedia2wikidata_trie=wikipedia2wikidata_trie,
        redirections=redirections,
        prefix=args.prefix
    )
    anchor_trie.save(join(args.out, 'trie.marisa'))
    (
        (
            trie_index2indices_offsets,
            trie_index2indices_values,
            trie_index2indices_counts
        ),
        (
            trie_index2contexts_offsets,
            trie_index2contexts_values,
            trie_index2contexts_counts
        )
    ) = construct_mapping(
        anchor_tags=args.anchor_tags,
        wikipedia2wikidata_trie=wikipedia2wikidata_trie,
        redirections=redirections,
        prefix=args.prefix,
        anchor_trie=anchor_trie
    )
    np.save(join(args.out, "trie_index2indices_offsets.npy"), trie_index2indices_offsets)
    np.save(join(args.out, "trie_index2indices_values.npy"), trie_index2indices_values)
    np.save(join(args.out, "trie_index2indices_counts.npy"), trie_index2indices_counts)

    np.save(join(args.out, "trie_index2contexts_offsets.npy"), trie_index2contexts_offsets)
    np.save(join(args.out, "trie_index2contexts_values.npy"), trie_index2contexts_values)
    np.save(join(args.out, "trie_index2contexts_counts.npy"), trie_index2contexts_counts)


if __name__ == "__main__":
    main()
