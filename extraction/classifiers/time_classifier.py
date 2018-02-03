"""
Create membership rules for entities based on their date of existence/birth/etc.
More classes can be created by selecting other key dates as hyperplanes.
"""
from numpy import (
    logical_and, logical_or, logical_not, logical_xor, where
)
from wikidata_linker_utils.logic import logical_negate, logical_ors, logical_ands
import wikidata_linker_utils.wikidata_properties as wprop


def wkp(c, name):
    """Convert a string wikipedia article name to its Wikidata index."""
    return c.article2id["enwiki/" + name][0][0]


def wkd(c, name):
    """Convert a wikidata QID to its wikidata index."""
    return c.name2index[name]


def classify(c):
    D1950 = 1950

    pre_1950 = logical_ors([
        c.attribute(wprop.PUBLICATION_DATE) < D1950,
        c.attribute(wprop.DATE_OF_BIRTH) < D1950,
        c.attribute(wprop.INCEPTION) < D1950,
        c.attribute(wprop.DISSOLVED_OR_ABOLISHED) < D1950,
        c.attribute(wprop.POINT_IN_TIME) < D1950,
        c.attribute(wprop.START_TIME) < D1950
    ])

    post_1950 = logical_and(logical_ors([
        c.attribute(wprop.PUBLICATION_DATE) >= D1950,
        c.attribute(wprop.DATE_OF_BIRTH) >= D1950,
        c.attribute(wprop.INCEPTION) >= D1950,
        c.attribute(wprop.DISSOLVED_OR_ABOLISHED) >= D1950,
        c.attribute(wprop.POINT_IN_TIME) >= D1950,
        c.attribute(wprop.START_TIME) >= D1950
    ]), logical_not(pre_1950))

    # some elements are neither pre 1950 or post 1950, they are "undated"
    # (e.g. no value was provided for any of the time attributes used
    # above)
    undated = logical_and(logical_not(pre_1950), logical_not(post_1950))
    print("%d items have no date information" % (undated.sum(),))
    return {
        "pre-1950": pre_1950,
        "post-1950": post_1950
    }
