import json
import msgpack
import bz2


def iterate_bytes_jsons(fin, batch_size=1000):
    current = []
    for l in fin:
        if l.startswith(b'{'):
            current.append(l)
        if len(current) >= batch_size:
            docs = json.loads('[' + b"".join(current).decode('utf-8').rstrip(',\n') + ']')
            for doc in docs:
                yield doc
            current = []
    if len(current) > 0:
        docs = json.loads('[' + b"".join(current).decode('utf-8').rstrip(',\n') + ']')
        for doc in docs:
            yield doc
        current = []


def iterate_text_jsons(fin, batch_size=1000):
    current = []
    for l in fin:
        if l.startswith('{'):
            current.append(l)
        if len(current) >= batch_size:
            docs = json.loads('[' + "".join(current).rstrip(',\n') + ']')
            for doc in docs:
                yield doc
            current = []
    if len(current) > 0:
        docs = json.loads('[' + "".join(current).rstrip(',\n') + ']')
        for doc in docs:
            yield doc
        current = []


def iterate_message_packs(fin):

    unpacker = msgpack.Unpacker(fin, encoding='utf-8', use_list=False)
    for obj in unpacker:
        yield obj


def open_wikidata_file(path, batch_size):
    if path.endswith('bz2'):
        with bz2.open(path, 'rb') as fin:
            for obj in iterate_bytes_jsons(fin, batch_size):
                yield obj
    elif path.endswith('json'):
        with open(path, 'rt') as fin:
            for obj in iterate_text_jsons(fin, batch_size):
                yield obj
    elif path.endswith('mp'):
        with open(path, 'rb') as fin:
            for obj in iterate_message_packs(fin):
                yield obj
    else:
        raise ValueError(
            "unknown extension for wikidata. "
            "Expecting bz2, json, or mp (msgpack)."
        )
