import sys
import importlib.util
import traceback

from os.path import basename, splitext


def reload_module(path):
    module_name, extension = splitext(basename(path))
    if extension != ".py":
        raise ValueError("path must have a .py extension (got %r)" % (path,))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enter_or_quit():
    wait = input("press any key to continue, q to quit.")
    received = wait.rstrip()
    if received == 'q':
        print("Bye.")
        sys.exit(0)
    else:
        return received


ALLOWED_RUNTIME_ERRORS = (
    TypeError,
    ValueError,
    IndexError,
    NameError,
    KeyError,
    AssertionError,
    AttributeError,
    ImportError,
    KeyboardInterrupt
)

ALLOWED_IMPORT_ERRORS = (
    SyntaxError,
    NameError,
    ImportError
)


def reload_run_retry(module_path, callback):
    while True:
        try:
            module = reload_module(module_path)
        except ALLOWED_IMPORT_ERRORS as e:
            print("issue reading %r, please fix." % (module_path,))
            print(str(e))
            traceback.print_exc(file=sys.stdout)
            enter_or_quit()
            continue
        try:
            result = callback(module)
        except ALLOWED_RUNTIME_ERRORS as e:
            print("issue running %r, please fix." % (module_path,))
            print(str(e))
            traceback.print_exc(file=sys.stdout)
            enter_or_quit()
            continue
        break
    return result
