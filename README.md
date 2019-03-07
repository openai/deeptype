**Status:** Archive (code is provided as-is, no updates expected)

DeepType: Multilingual Entity Linking through Neural Type System Evolution
--------------------------------------------------------------------------

This repository contains code necessary for designing, evolving type systems, and training neural type systems. To read more about this technique and our results [see this blog post](https://blog.openai.com/discovering-types-for-entity-disambiguation/) or [read the paper](https://arxiv.org/abs/1802.01021).

Authors: Jonathan Raiman & Olivier Raiman

Our latest approach to learning symbolic structures from data allows us to discover a set of task specific constraints on a neural network in the form of a type system, to guide its understanding of documents, and obtain state of the art accuracy at [recognizing entities in natural language](https://en.wikipedia.org/wiki/Entity_linking). Recognizing entities in documents can be quite challenging since there are often millions of possible answers. However, when using a type system to constrain the options to only those that semantically "type check," we shrink the answer set and make the problem dramatically easier to solve. Our new results suggest that learning types is a very strong signal for understanding natural language: if types were given to us by an oracle, we find that it is possible to obtain accuracies of 98.6-99% on two benchmark tasks [CoNLL (YAGO)](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/) and the [TAC KBP 2010 challenge](https://pdfs.semanticscholar.org/b7fb/11ef06b0dcdc89ef0a5507c6c9ccea4206d8.pdf).

### Data collection

Get wikiarticle -> wikidata mapping (all languages) + Get anchor tags, redirections, category links, statistics (per language). To store all wikidata ids, their key properties (`instance of`, `part of`, etc..), and
a mapping from all wikipedia article names to a wikidata id do as follows,
along with wikipedia anchor tags and links, in three languages: English (en), French (fr), and Spanish (es):

```
export DATA_DIR=data/
./extraction/full_preprocess.sh ${DATA_DIR} en fr es
```

### Create a type system manually and check oracle accuracy:

To build a graph projection using a set of rules inside `type_classifier.py`
(or any Python file containing a `classify` method), and a set of nodes
that should not be traversed in `blacklist.json`:

```
export LANGUAGE=fr
export DATA_DIR=data/
python3 extraction/project_graph.py ${DATA_DIR}wikidata/ extraction/classifiers/type_classifier.py
```

To save a graph projection as a numpy array along with a list of classes to a
directory stored in `CLASSIFICATION_DIR`:

```
export LANGUAGE=fr
export DATA_DIR=data/
export CLASSIFICATION_DIR=data/type_classification
python3 extraction/project_graph.py ${DATA_DIR}wikidata/ extraction/classifiers/type_classifier.py  --export_classification ${CLASSIFICATION_DIR}
```

To use the saved graph projection on wikipedia data to test out how discriminative this
classification is (Oracle performance) (edit the config file to make changes to the classification used):

```
export DATA_DIR=data/
python3 extraction/evaluate_type_system.py extraction/configs/en_disambiguator_config_export_small.json --relative_to ${DATA_DIR}
```

### Obtain learnability scores for types

```bash
export DATA_DIR=data/
python3 extraction/produce_wikidata_tsv.py extraction/configs/en_disambiguator_config_export_small.json --relative_to ${DATA_DIR} sample_data.tsv
python3 learning/evaluate_learnability.py sample_data.tsv --out report.json --wikidata ${DATA_DIR}wikidata/
```

See `learning/LearnabilityStudy.ipynb` for a visual analysis of the AUC scores.

### Evolve a type system

```bash
python3 extraction/evolve_type_system.py extraction/configs/en_disambiguator_config_export_small.json --relative_to ${DATA_DIR}  --method cem  --penalty 0.00007
```
Method can be `cem`, `greedy`, `beam`, or `ga`, and penalty is the soft constraint on the size of the type system (lambda in the paper).

#### Convert a type system solution into a trainable type classifier

The output of `evolve_type_system.py` is a set of types (root + relation) that can be used to build a type system. To create a config file that can be used to train an LSTM use the jupyter notebook `extraction/TypeSystemToNeuralTypeSystem.ipynb`.

### Train a type classifier using a type system

For each language create a training file:

```
export LANGUAGE=en
python3 extraction/produce_wikidata_tsv.py extraction/configs/${LANGUAGE}_disambiguator_config_export.json /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.tsv  --relative_to /Volumes/Samsung_T3/tahiti/2017-12/
```

Then create an H5 file from each language containing the mapping from tokens to their entity ids in Wikidata:

```
export LANGUAGE=en
python3 extraction/produce_windowed_h5_tsv.py  /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.tsv /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.h5 /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_dev.h5 --window_size 10  --validation_start 1000000 --total_size 200500000
```

Create a training config with all languages, `my_config.json`. Paths to the datasets is relative to config file (e.g. you can place it in the same directory as the dataset h5 files):
[Note: set `wikidata_path` to where you extracted wikidata information, and `classification_path` to where you exported the classifications with `project_graph.py`]. See learning/configs for a pre written config covering English, French, Spanish, German, and Portuguese.

```
{
    "datasets": [
        {
            "type": "train",
            "path": "en_train.h5",
            "x": 0,
            "ignore": "other",
            "y": [
                {
                    "column": 1,
                    "objective": "type",
                    "classification": "type_classification"
                },...
            ],
            "ignore": "other",
            "comment": "#//#"
        },
        {
            "type": "dev",
            "path": "en_dev.h5",
            "x": 0,
            "ignore": "other",
            "y": [
                {
                    "column": 1,
                    "objective": "type",
                    "classification": "type_classification"
                },...
            ],
            "ignore": "other",
            "comment": "#//#"
        }, ...
    ],
    "features": [
        {
            "type": "word",
            "dimension": 200,
            "max_vocab": 1000000
        },...
    ],
    "objectives": [
        {
            "name": "type",
            "type": "softmax",
            "vocab": "type_classes.txt"
        }, ...
    ],
    "wikidata_path": "wikidata",
    "classification_path": "classifications"
}
```

Launch training on a single gpu:

```
CUDA_VISIBLE_DEVICES=0 python3 learning/train_type.py my_config.json --cudnn --fused --hidden_sizes 200 200 --batch_size 256 --max_epochs 10000  --name TypeClassifier --weight_noise 1e-6  --save_dir my_great_model  --anneal_rate 0.9999
```

Several key parameters:

- `name`: main scope for model variables, avoids name clashing when multiple classifiers are loaded
- `batch_size`: how many examples are used for training simultaneously, can cause out of memory issues
- `max_epochs`: length of training before auto-stopping. In practice this number should be larger than needed.
- `fused`: glue all output layers into one, and do a single matrix multiply (recommended).
- `hidden_sizes`: how many stacks of LSTMs are used, and their sizes (here 2, each with 200 dimensions).
- `cudnn`: use faster CuDNN kernels for training
- `anneal_rate`: shrink the learning rate by this amount every 33000 training steps
- `weight_noise`: sprinkle Gaussian noise with this standard deviation on the weights of the LSTM (regularizer, recommended).


#### To test that training works:

You can test that training works as expected using the dummy training set containing a Part of Speech CRF objective and cat vs dogs log likelihood objective is contained under learning/test:

```bash
python3 learning/train_type.py learning/test/config.json
```

### Installation

#### Mac OSX

```
pip3 install -r requirements.txt
pip3 install wikidata_linker_utils_src/
```

#### Fedora 25

```
sudo dnf install redhat-rpm-config
sudo dnf install gcc-c++
sudo pip3 install marisa-trie==0.7.2
sudo pip3 install -r requirements.txt
pip3 install wikidata_linker_utils_src/
```
