Guide to Human Type Classification
----------------------------------

For each type dimension you want, a method for projecting the wikidata type system into
distinct classes is needed.

# Usage

## Defining a Graph projection

First create a Python script that has a single method `classify` which returns a dictionary
of numpy boolean arrays:

```python
import wikidata_linker_utils.wikidata_properties as wprop

def classify(c):
    pre_1950 = c.attribute(wprop.DATE_OF_BIRTH) < 1950
    return {"pre-1950": pre_1950}
```

Note: Anything that isn't "True" in the array will become part of the `Other` class.

You can generate this classification as follows, by first saving your script to a
file (say `my_classifier.py`), then callling `extraction/project_graph.py`:

```bash
WIKIDATA_PATH=/Volumes/Samsung_T3/tahiti/2017-02/wikidata
python3 extraction/project_graph.py $WIKIDATA_PATH my_classifier.py
```

This will then output information about the projection, including who was left out,
included, and some general statistics.

You can see other examples under `extraction/classifiers`.

In particular many use `np.logical_or`, `np.logical_and`, etc... to define richer
inheritance rules.

Wikidata can be also be accessed using both "attributes" (e.g. dates),
and "relations" (e.g. "instance of"). You can then produce a boolean array
from relations as follows, using the relation "instance of":

```python
def wkp(c, name):
    """Convert a human-readable Wikipedia name (from enwiki)
    into its numeric wikidata id."""
    return c.article2id['enwiki/' + name][0][0]

def classify(c):
    HUMAN = wkp(c, "Human")
    is_human = c.satisfy([wprop.INSTANCE_OF], [HUMAN])
    return {"human": is_human}
```

### Tooling & debugging

Within the `classify` function, you can make different calls to print out,
or otherwise report on the behavior of your system. However it is sometimes
hard to know in a rich way the neighborhood or statistics of a particular
projection. Several methods help report on the boolean arrays in a human
readable way:

#### Class Report

The method `TypeCollection.class_report` provides info about the number of
unique children, items related by some edge type (see example below).

```python
c.class_report([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], is_human, name="Human")
```

Note: We can answer "what are the most popular pages/articles in
this boolean array?" if the `language_path` argument is given to `extraction/project_graph.py`.
The  then Wikipedia article fixed points weighted by how often they are used as links
in wikipedia are printed.

However, if the boolean array deals with a different
language than the language_path, or is somehow more niche, then this printing
may not print many items despite its popularity in a different country/subculture.

#### Describe Connection

Sometimes two nodes are connected even though this connection is unexpected
(e.g. Toaster (Q14890) and Science (Q336)). To understand what edges can be used
to connect from one node to the other the `TypeCollection.describe_connection`
method prints the path:

```python
c.describe_connection("Q14890", "Q336", [wprop.INSTANCE_OF, wprop.SUBCLASS_OF)
```

####

### Exporting

Call `extraction/project_graph` as you did previously and add the following argument `export_classification`:

```bash
WIKIDATA_PATH=/Volumes/Samsung_T3/tahiti/2017-02/wikidata
python3 extraction/project_graph.py $WIKIDATA_PATH my_classifier.py  --export_classification /path/to/my_classification
```

### Blacklist

Several nodes can cause issues when doing graph projections because they are too generic or otherwise unhelpful.
A remedy is to state that certain edges cannot be crossed.

We can do this using the `extraction/blacklist.json` file. This file is a JSON dictionary with two keys:
`bad_node` and `bad_node_pair`. `bad_node` contains singleton vertices that should be ignored. While `bad_node_pair` denotes a transition that is forbidden.

Note: `bad_node_pair` is not relation specific, and thus it will cancel out any connection between node pairs.

### Interactivity

The `extraction/project_graph` script acts like a REPL. Each time an error is found in the script, or after the script
is run, the script gets reloaded. You can therefore iterate on the script, hit ENTER, and try it again. Each
time it will export the classification if it ran without errors.

The blacklist is reloaded on each REPL run in interactive mode.

### Options

- Use `--language_path` to specify the location of a wikipedia trie.
- Use `--num_names_to_load` to control whether names are pre-loaded (longer start time), but faster to do reporting (e.g. to show membership examples).

## Testing against a corpus

To test the effectiveness of the classification you can use a sample from Wikipedia to benchmark classification using the `extraction/evaluate_type_system.py`. First create a config file that describes your classifiers & paths:

```json
{
    "wiki": "enwiki-latest-pages-articles.xml",
    "wikidata": "wikidata",
    "prefix": "enwiki",
    "sample_size": 1000,
    "num_names_to_load": 4000,
    "language_path": "en_trie_fixed",
    "redirections": "en_redirections.tsv",
    "classification": [
        "/path/to/my_classification"
    ]
}
```

You can now call it as follows, after naming your config file `my_config.json`:

```
BASE_PATH=/Volumes/Samsung_T3/tahiti/2017-02/
python3 extraction/evaluate_type_system.py my_config.json --relative_to $BASE_PATH
```

#### Config notes

- `BASE_PATH` points to the parent directory for "wiki", "wikidata", and "language_path" (or if these paths are made absolute, then `--relative_to` is not needed).
- in the config "prefix" specifies what language/lookup method should be used when parsing the "wiki" file. If you switch to French, "frwiki-latest-pages-articles.xml", then use `"prefix": "frwiki"` instead.
- redirections is another file collected for each wikipedia corpus. Use the redirections corresponding to the wikipedia dump you are using (so `fr_redirections.tsv` for French).

### Interactivity

After running this function you can now update your exports and run it again by hitting ENTER.

### Options

- Use `--noverbose` to not see specific ambiguity examples, only stats.
