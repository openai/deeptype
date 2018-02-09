import json

from collections import namedtuple
from os.path import join, dirname


def dict_fix_relative_paths(basepath, relative_paths):
    if relative_paths is None:
        relative_paths = []

    def load(d):
        new_obj = d.copy()
        for key in relative_paths:
            if key in new_obj:
                if isinstance(new_obj[key], str):
                    new_obj[key] = join(basepath, new_obj[key])
                elif isinstance(new_obj[key], list) and len(new_obj[key]) > 0 and isinstance(new_obj[key][0], str):
                    new_obj[key] = [join(basepath, path) for path in new_obj[key]]
        return new_obj
    return load


def load_config(path, relative_paths=None, defaults=None, relative_to=None):
    if relative_to is None:
        relative_to = dirname(path)
    object_hook = dict_fix_relative_paths(relative_to, relative_paths)
    with open(path, "rt") as fin:
        obj = json.load(
            fin,
            object_hook=object_hook
        )
    if defaults is not None:
        for key, value in defaults.items():
            if key not in obj:
                obj[key] = value
    return json.loads(
        json.dumps(obj),
        object_hook=lambda d: namedtuple('X', d.keys())(*d.values())
    )


def json_loads(bytes):
    return json.loads(bytes.decode('utf-8'))


def json_serializer(x):
    return json.dumps(
        x, check_circular=False, separators=(',', ':')
    ).encode('utf-8')
