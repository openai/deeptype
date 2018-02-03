"""
Compress a jsonl version of Wikidata by throwing about descriptions
and converting file to msgpack format.

Usage
-----

```
python3 compress_wikidata_msgpack.py wikidata.json wikidata.msgpack
```

"""
import argparse
import msgpack

from wikidata_linker_utils.wikidata_iterator import open_wikidata_file
from wikidata_linker_utils.progressbar import get_progress_bar


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('wikidata')
    parser.add_argument('out')
    return parser.parse_args(args=args)


def main():
    args = parse_args()
    approx_max_quantity = 24642416
    pbar = get_progress_bar('compress wikidata', max_value=approx_max_quantity, item='entities')
    pbar.start()
    seen = 0
    with open(args.out, "wb") as fout:
        for doc in open_wikidata_file(args.wikidata, 1000):
            seen += 1
            if 'descriptions' in doc:
                del doc['descriptions']
            if 'labels' in doc:
                del doc['labels']
            if 'aliases' in doc:
                del doc['aliases']
            for claims in doc['claims'].values():
                for claim in claims:
                    if 'id' in claim:
                        del claim['id']
                    if 'rank' in claim:
                        del claim['rank']
                    if 'references' in claim:
                        for ref in claim['references']:
                            if 'hash' in ref:
                                del ref['hash']
                    if 'qualifiers' in claim:
                        for qualifier in claim['qualifiers'].values():
                            if 'hash' in qualifier:
                                del qualifier['hash']
            fout.write(msgpack.packb(doc))
            if seen % 1000 == 0:
                if seen < approx_max_quantity:
                    pbar.update(seen)
    pbar.finish()


if __name__ == "__main__":
    main()
