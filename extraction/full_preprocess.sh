#!/bin/bash

# stop script on error and print it
set -e
# inform me of undefined variables
set -u
# handle cascading failures well
set -o pipefail

function ensure_dir {
    if [ "${1: -1}" != "/" ]; then
        echo "${1}/"
    else
        echo $1
    fi
}

if [ "$#" -lt 2 ]; then
    echo "Not enough arguments."
    echo "Usage: $0 DATA_DIR LANGUAGE [LANGUAGE ...]"
    exit
fi

DATA_DIR=$1
DATA_DIR="$(ensure_dir $DATA_DIR)"
echo "Downloading wikidata into ${DATA_DIR}."
LANGUAGES=${@:2}

for LANGUAGE in $LANGUAGES
do
    echo "Will prepare language: ${LANGUAGE}"
done

echo "Creating data directory"
mkdir -p $DATA_DIR
echo "Done."

echo "Downloading and preparing Wikidata:"
if [ -f "${DATA_DIR}latest-all.mp" ]
then
    echo "Already compressed Wikidata"
else
    if [ -f "${DATA_DIR}latest-all.json" ]
    then
        echo "Already decompressed Wikidata"
        python3 extraction/compress_wikidata_msgpack.py ${DATA_DIR}latest-all.json ${DATA_DIR}latest-all.mp
        rm ${DATA_DIR}latest-all.json
    else
        if [ -f "${DATA_DIR}latest-all.json.bz2" ]
        then
            echo "Already downloaded ${DATA_DIR}latest-all.json.bz2"
            bzip2 -d -v4 ${DATA_DIR}latest-all.json.bz2
            python3 extraction/compress_wikidata_msgpack.py ${DATA_DIR}latest-all.json ${DATA_DIR}latest-all.mp
            rm ${DATA_DIR}latest-all.json
        else
            wget -O ${DATA_DIR}latest-all.json.bz2 https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
            bzip2 -d -v4 ${DATA_DIR}latest-all.json.bz2
            python3 extraction/compress_wikidata_msgpack.py ${DATA_DIR}latest-all.json ${DATA_DIR}latest-all.mp
            rm ${DATA_DIR}latest-all.json
        fi
    fi
fi
mkdir -p ${DATA_DIR}wikidata/
python3 extraction/get_wikiname_to_wikidata.py ${DATA_DIR}latest-all.mp ${DATA_DIR}wikidata/
echo "Done with wikidata."

for LANGUAGE in $LANGUAGES
do
    echo "Preparing language: ${LANGUAGE}"
    if [ -f "${DATA_DIR}${LANGUAGE}wiki-latest-pages-articles.xml" ]
    then
        echo "Already downloaded and extracted ${LANGUAGE}wiki-latest-pages-articles.xml."
    else
        wget -O ${DATA_DIR}${LANGUAGE}wiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/${LANGUAGE}wiki/latest/${LANGUAGE}wiki-latest-pages-articles.xml.bz2
        bzip2 -d -v4 ${DATA_DIR}${LANGUAGE}wiki-latest-pages-articles.xml.bz2
    fi
    python3 extraction/get_redirection_category_links.py ${DATA_DIR}${LANGUAGE}wiki-latest-pages-articles.xml ${DATA_DIR}${LANGUAGE}_anchors.tsv ${DATA_DIR}${LANGUAGE}_redirections.tsv ${DATA_DIR}${LANGUAGE}_category_links.tsv
    python3 extraction/convert_category_links_to_wikidata.py ${DATA_DIR}wikidata/wikititle2wikidata.marisa ${DATA_DIR}wikidata/wikidata_ids.txt ${LANGUAGE}wiki ${DATA_DIR}${LANGUAGE}_category_links.tsv ${DATA_DIR}wikidata/
    python3 extraction/convert_anchor_tags_to_wikidata.py ${DATA_DIR}wikidata/wikititle2wikidata.marisa ${LANGUAGE}wiki ${DATA_DIR}${LANGUAGE}_anchors.tsv ${DATA_DIR}${LANGUAGE}_redirections.tsv ${DATA_DIR}${LANGUAGE}_trie
    # apply link fixing strategy here:
    python3 extraction/fast_link_fixer.py ${DATA_DIR}wikidata ${DATA_DIR}${LANGUAGE}_trie ${DATA_DIR}${LANGUAGE}_trie_fixed
done
