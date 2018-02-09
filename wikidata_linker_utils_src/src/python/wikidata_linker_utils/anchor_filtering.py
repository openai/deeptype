import re

STOP_WORDS = {'a', 'an', 'in', 'the', 'of', 'it', 'from', 'with', 'this', 'that', 'they', 'he',
              'she', 'some', 'where', 'what', 'since', 'his', 'her', 'their', 'le', 'la', 'les', 'il',
              'elle', 'ce', 'ça', 'ci', 'ceux', 'ceci', 'cela', 'celle', 'se', 'cet', 'cette',
              'dans', 'avec', 'con', 'sans', 'pendant', 'durant', 'avant', 'après', 'puis', 'el', 'lo', 'la',
              'ese', 'esto', 'que', 'qui', 'quoi', 'dont', 'ou', 'où', 'si', 'este', 'esta', 'cual',
              'eso', 'ella', 'depuis', 'y', 'a', 'à', 'su', 'de', "des", 'du', 'los', 'las', 'un', 'une', 'una',
              'uno', 'para', 'asi', 'later', 'into', 'dentro', 'dedans', 'depuis', 'después', 'desde',
              'al', 'et', 'por', 'at', 'for', 'when', 'why', 'how', 'with', 'whether', 'if',
              'thus', 'then', 'and', 'but', 'on', 'during', 'while', 'as', 'within', 'was', 'is',
              'est', 'au', 'fait', 'font', 'va', 'vont', 'sur', 'en', 'pour', 'del', 'cuando',
              'cuan', 'do', 'does', 'until', 'sinon', 'encore', 'to', 'by', 'be', 'which',
              'have', 'not', 'were', 'has', 'also', 'its', 'isbn', 'pp.', "&amp;", "p.", 'ces', 'o'}


def starts_with_apostrophe_letter(word):
    return (
        word.startswith("l'") or
        word.startswith("L'") or
        word.startswith("d'") or
        word.startswith("D'") or
        word.startswith("j'") or
        word.startswith("J'") or
        word.startswith("t'") or
        word.startswith("T'")
    )


PUNCTUATION = {"'", ",", "-", "!", ".", "?", ":", "’"}


def clean_up_trie_source(source, lowercase=True):
    source = source.rstrip().strip('()[]')
    if len(source) > 0 and (source[-1] in PUNCTUATION or source[0] in PUNCTUATION):
        return ""
    # remove l'
    if starts_with_apostrophe_letter(source):
        source = source[2:]
    if source.endswith("'s"):
        source = source[:-2]
    tokens = source.split()
    while len(tokens) > 0 and tokens[0].lower() in STOP_WORDS:
        tokens = tokens[1:]
    while len(tokens) > 0 and tokens[-1].lower() in STOP_WORDS:
        tokens = tokens[:-1]
    joined_tokens = " ".join(tokens)
    if lowercase:
        return joined_tokens.lower()
    return joined_tokens


ORDINAL_ANCHOR = re.compile("^\d+(st|th|nd|rd|er|eme|ème|ère)$")
NUMBER_PUNCTUATION = re.compile("^\d+([\/\-,\.:;%]\d*)+$")


def anchor_is_ordinal(anchor):
    return ORDINAL_ANCHOR.match(anchor) is not None


def anchor_is_numbers_slashes(anchor):
    return NUMBER_PUNCTUATION.match(anchor) is not None


def acceptable_anchor(anchor, anchor_trie, blacklist=None):
    return (len(anchor) > 0 and
            not anchor.isdigit() and
            not anchor_is_ordinal(anchor) and
            not anchor_is_numbers_slashes(anchor) and
            anchor in anchor_trie and
            (blacklist is None or anchor not in blacklist))
