import subprocess


def execute_bash(command):
    """
    Executes bash command, prints output and
    throws an exception on failure.
    """
    process = subprocess.Popen(command,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    assert process.returncode == 0


def get_bash_result(command):
    """
    Executes bash command, returns output and throws
    an exception on failure.
    """
    process = subprocess.Popen(command,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
    out = [line for line in process.stdout]
    process.wait()
    assert process.returncode == 0
    return out


def count_lines(path):
    return int(
        get_bash_result('wc -l %s' % (path,))[0].strip().split(' ')[0]
    )
