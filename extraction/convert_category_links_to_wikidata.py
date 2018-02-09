import argparse
import marisa_trie

import numpy as np

from os.path import join

from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.bash import count_lines
from wikidata_linker_utils.offset_array import save_record_with_offset


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("wikipedia2wikidata_trie",
        help="Location of wikipedia -> wikidata mapping trie.")
    parser.add_argument("wikidata_ids")
    parser.add_argument("prefix")
    parser.add_argument("category_links")
    parser.add_argument("out")
    return parser.parse_args(argv)

def main():
    args = parse_args()
    trie = marisa_trie.RecordTrie('i').load(args.wikipedia2wikidata_trie)
    print('loaded trie')

    num_lines = count_lines(args.category_links)
    num_ids = count_lines(args.wikidata_ids)
    missing = []
    num_missing = 0
    num_broken = 0
    all_category_links = [[] for i in range(num_ids)]
    with open(args.category_links, 'rt') as fin:
        fin_pbar = get_progress_bar('reading category_links', max_value=num_lines)(fin)
        for line in fin_pbar:
            try:
                origin, dest = line.rstrip('\n').split('\t')
            except:
                num_broken += 1
                continue
            if len(dest) == 0:
                num_broken += 1
                continue
            origin = args.prefix + '/' + origin
            prefixed_dest = args.prefix + '/' + dest
            origin_index = trie.get(origin, None)
            dest_index = trie.get(prefixed_dest, None)

            if dest_index is None:
                prefixed_dest = args.prefix + '/' + dest[0].upper() + dest[1:]
                dest_index = trie.get(prefixed_dest, None)

            if origin_index is None or dest_index is None:
                missing.append((origin, prefixed_dest))
                num_missing += 1
            else:
                all_category_links[origin_index[0][0]].append(dest_index[0][0])

    print("%d/%d category links could not be found in wikidata" % (num_missing, num_lines))
    print("%d/%d category links were malformed" % (num_broken, num_lines))
    print("Missing links sample:")
    for origin, dest in missing[:10]:
        print("%r -> %r" % (origin, dest))
    save_record_with_offset(
        join(args.out, "wikidata_%s_category_links" % (args.prefix,)),
        all_category_links
    )


if __name__ == "__main__":
    main()
