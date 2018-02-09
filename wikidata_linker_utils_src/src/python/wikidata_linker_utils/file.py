from os.path import exists
from os import stat


def true_exists(fname):
    return exists(fname) and stat(fname).st_size > 100
