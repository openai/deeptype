"""
Perform a reduction on the anchors to articles relation
by finding different articles refering to the same item
and making the anchor point to the most common version,
or by using the wikidata graph to find instance of, and
other parent-child relations that allow one article to
encompass or be more generic than its co-triggerable
articles.

Usage:
------

```
DATA_DIR=data/wikidata
LANG_DIR=data/en_trie
FIXED_LANG_DIR=data/en_trie_fixed
python3 fast_link_fixer.py ${WIKIDATA_PATH} ${LANG_DIR} ${FIXED_LANG_DIR}
```
"""
import argparse
import time
import shutil

from os.path import join, realpath, dirname
from os import makedirs

import numpy as np
import marisa_trie

from wikidata_linker_utils.type_collection import get_name, TypeCollection
from wikidata_linker_utils.logic import logical_and, logical_ands, logical_not, logical_or, logical_ors
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.offset_array import OffsetArray
from wikidata_linker_utils.file import true_exists
import wikidata_linker_utils.wikidata_properties as wprop

from wikidata_linker_utils.successor_mask import (
    related_promote_highest, extend_relations, reduce_values,
    remap_offset_array
)

SCRIPT_DIR = dirname(realpath(__file__))

from numpy import logical_not, logical_or, logical_and
from wikidata_linker_utils.logic import logical_ors
IS_HISTORY = None
IS_PEOPLE = None
IS_BREED = None
IS_PEOPLE_GROUP = None
IS_LIST_ARTICLE = None
IS_LANGUAGE_ALPHABET = None
IS_SPORTS_TEAM = None
IS_CARDINAL_DIRECTION = None
IS_POLITICAL_PARTY = None
IS_SOCIETY = None
IS_POSITION = None
IS_CHARACTER_HUMAN = None
IS_POLITICAL_ORGANIZATION = None
IS_LANDFORM = None
IS_THING = None
IS_BATTLE = None
IS_EVENT = None
IS_ACTIVITY = None
IS_THOROUGHFARE = None
IS_KINSHIP = None
IS_EPISODE_LIST = None

def wkp(c, name):
    return c.article2id['enwiki/' + name][0][0]

def wkd(c, name):
    return c.name2index[name]

def initialize_globals(c):
    """global variables that guide the metonymy/anaphora removal process."""
    global IS_HISTORY
    global IS_PEOPLE
    global IS_PEOPLE_GROUP
    global IS_LIST_ARTICLE
    global IS_COUNTRY
    global IS_BREED
    global IS_EVENT_SPORT
    global IS_LANGUAGE_ALPHABET
    global IS_SPORTS_TEAM
    global IS_CARDINAL_DIRECTION
    global IS_ACTIVITY
    global IS_POLITICAL_PARTY
    global IS_SOCIETY
    global IS_BATTLE
    global IS_POSITION
    global IS_LANDFORM
    global IS_CHARACTER_HUMAN
    global IS_POLITICAL_ORGANIZATION
    global IS_THING
    global IS_THOROUGHFARE
    global IS_EVENT
    global IS_KINSHIP
    global IS_EPISODE_LIST
    PEOPLE = wkd(c, "Q2472587")
    NATIONALITY = wkd(c, "Q231002")
    ASPECT_OF_HIST = wkd(c, "Q17524420")
    HISTORY = wkd(c, "Q309")
    LIST_ARTICLE = wkd(c, "Q13406463")
    WAR = wkd(c, "Q198")
    COUNTRY = wkd(c, "Q6256")
    FORMER_COUNTRY = wkd(c, "Q3024240")
    DOMINION = wkd(c, "Q223832")
    LANGUAGE = wkd(c, "Q34770")
    ALPHABET = wkd(c, "Q9779")
    COLONY = wkd(c, "Q133156")
    GOVERNORATE = wkd(c, "Q1798622")
    SPORTS_TEAM = wkd(c, "Q12973014")
    ATHLETIC_CONFERENCE = wkd(c, "Q2992826")
    CARDINAL_DIRECTION = wkd(c, "Q23718")
    POLITICAL_PARTY = wkd(c, "Q7278")
    STATE = wkd(c, "Q7275")
    DYNASTY = wkd(c, "Q164950")
    SOCIETY = wkd(c, "Q8425")
    MENS_SINGLES = wkd(c, "Q16893072")
    SPORT = wkd(c, "Q349")
    POSITION = wkd(c, "Q4164871")
    HUMAN = wkd(c, "Q5")
    FICTIONAL_CHARACTER = wkd(c, "Q95074")
    BREED = wkd(c, "Q38829")
    ORTHOGRAPHY = wkd(c, "Q43091")
    POLITICAL_ORGANIZATION = wkd(c, "Q7210356")
    GROUP_OF_HUMANS = wkd(c, "Q16334295")
    LANDFORM = wkd(c, "Q271669")
    BATTLE = wkd(c, "Q178561")
    FOOD = wkd(c, "Q2095")
    DRINK = wkd(c, "Q40050")
    ANIMAL = wkd(c, "Q16521")
    WORK = wkd(c, "Q386724")
    AUTOMOBILE_MODEL = wkd(c, "Q3231690")
    GOOD = wkd(c, "Q28877")
    VEHICLE = wkd(c, "Q42889")
    PUBLICATION = wkd(c, "Q732577")
    AUDIOVISUAL = wkd(c, "Q2431196")
    TERRITORIAL_ENTITY = wkd(c, "Q15642541")
    GEOGRAPHIC_OBJECT = wkd(c, "Q618123")
    ASTRO_OBJECT = wkd(c, "Q17444909")
    EVENT_SPORTING = wkd(c, "Q1656682")
    EVENT_OCCURRENCE = wkd(c, "Q1190554")
    ELECTROMAGNETIC_SPECTRUM = wkd(c, "Q133139")
    MAGICAL_ORG = wkd(c, "Q14946195")
    AUTONOM_CHURCH = wkd(c, "Q20871948")
    SIGN = wkd(c, "Q3695082")
    FORM_OF_GOVERNMENT = wkd(c, "Q1307214")
    SPORTS_ORG = wkd(c, "Q4438121")
    RECURRING_SPORTING_EVENT = wkd(c, "Q18608583")
    CLASS_SCHEME = wkd(c, "Q5962346")
    STYLE = wkd(c, "Q1292119")
    SIGN_SYSTEM = wkd(c, "Q7512598")
    PHYSICAL_PHENOMENON = wkd(c, "Q1293220")
    LAW = wkd(c, "Q7748")
    WATERCOURSE = wkd(c, "Q355304")
    BODY_OF_WATER = wkd(c, "Q15324")
    CHEMICAL_SUBSTANCE = wkd(c, "Q79529")
    HISTORICAL_PERIOD = wkd(c, "Q11514315")
    ACTIVITY = wkd(c, "Q815962")
    THOROUGHFARE = wkd(c, "Q83620")
    KINSHIP = wkd(c, "Q171318")
    FICTIONAL_HUMAN = wkd(c, "Q15632617")
    EPISODE = wkd(c, "Q1983062")

    IS_CHARACTER_HUMAN = c.satisfy(
        [wprop.INSTANCE_OF, wprop.SUBCLASS_OF, wprop.IS_A_LIST_OF],
        [HUMAN, FICTIONAL_HUMAN, FICTIONAL_CHARACTER]
    )
    # to be a history you must be an aspect of history
    # but not a history itself:
    IS_HISTORY = logical_and(
        c.satisfy([wprop.INSTANCE_OF], [ASPECT_OF_HIST]),
        logical_not(c.satisfy([wprop.INSTANCE_OF], [HISTORY]))
    )
    IS_PEOPLE = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [PEOPLE, NATIONALITY])
    IS_PEOPLE_GROUP = np.logical_or(
        IS_PEOPLE,
        c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [GROUP_OF_HUMANS, MAGICAL_ORG, AUTONOM_CHURCH])
    )
    IS_LIST_ARTICLE = c.satisfy([wprop.INSTANCE_OF], [LIST_ARTICLE])
    IS_LANGUAGE_ALPHABET = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
        [LANGUAGE, ALPHABET, ORTHOGRAPHY, SIGN_SYSTEM]
    )
    IS_COUNTRY = c.satisfy([wprop.INSTANCE_OF], [COUNTRY, FORMER_COUNTRY, DOMINION, COLONY, STATE, DYNASTY, GOVERNORATE])
    IS_SPORTS_TEAM = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF, wprop.PART_OF], [SPORTS_TEAM, ATHLETIC_CONFERENCE, SPORTS_ORG, RECURRING_SPORTING_EVENT])
    IS_CARDINAL_DIRECTION = c.satisfy([wprop.INSTANCE_OF], [CARDINAL_DIRECTION])
    IS_POLITICAL_PARTY = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [POLITICAL_PARTY])
    IS_SOCIETY = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [SOCIETY, HISTORICAL_PERIOD])
    IS_POSITION = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [POSITION])
    IS_BREED = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [BREED])
    IS_POLITICAL_ORGANIZATION = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [POLITICAL_ORGANIZATION, FORM_OF_GOVERNMENT])
    IS_LANDFORM = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [LANDFORM, TERRITORIAL_ENTITY, GEOGRAPHIC_OBJECT, ASTRO_OBJECT, WATERCOURSE, BODY_OF_WATER])
    IS_EVENT_SPORT = c.satisfy([wprop.SUBCLASS_OF, wprop.PART_OF, wprop.INSTANCE_OF], [EVENT_SPORTING, SPORT])
    IS_THING = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
        [
            AUTOMOBILE_MODEL,
            FOOD,
            DRINK,
            STYLE,
            ANIMAL,
            GOOD,
            LAW,
            CHEMICAL_SUBSTANCE,
            SIGN,
            VEHICLE,
            PHYSICAL_PHENOMENON,
            PUBLICATION,
            AUDIOVISUAL,
            CLASS_SCHEME,
            WORK,
            ELECTROMAGNETIC_SPECTRUM
        ]
    )
    IS_THOROUGHFARE = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [THOROUGHFARE])
    IS_ACTIVITY = c.satisfy([wprop.INSTANCE_OF], [ACTIVITY])
    IS_EVENT = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [EVENT_OCCURRENCE])
    IS_BATTLE = c.satisfy([wprop.SUBCLASS_OF, wprop.INSTANCE_OF], [BATTLE])
    IS_KINSHIP = c.satisfy([wprop.INSTANCE_OF], [KINSHIP])
    IS_EPISODE_LIST = c.satisfy([wprop.IS_A_LIST_OF], [EPISODE])


def get_relation_data(collection, relation_paths):
    """Prepare relations for usage inside extend_relations."""
    out = []
    for path in relation_paths:
        promote = path.get("promote", False)
        numpy_path = []
        for step in path["steps"]:
            if isinstance(step, str):
                step_name, max_usage = step, 1
            else:
                step_name, max_usage = step
            relation = collection.relation(step_name)
            numpy_path.append((relation.offsets, relation.values, max_usage))
        inv_relation = collection.get_inverted_relation(step_name).edges() > 0
        out.append((numpy_path, inv_relation, promote))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("wikidata")
    parser.add_argument("language_path")
    parser.add_argument("new_language_path")
    parser.add_argument("--steps", type=int, default=3,
        help="how many time should fixing be recursed (takes "
             "about 2mn per step. Has diminishing returns).")
    return parser.parse_args()


def get_trie_properties(trie, offsets, values):
    """Obtain the length of every trigger in the trie."""
    anchor_length = np.zeros(len(values), dtype=np.int32)
    start, end = 0, 0
    for idx, key in enumerate(trie.iterkeys()):
        end = offsets[idx]
        anchor_length[start:end] = len(key)
        start = end
    return anchor_length


def fix(collection,
        offsets,
        values,
        counts,
        anchor_length,
        num_category_link=8,
        keep_min=5):
    relations_that_can_extend = [
        {"steps": [wprop.INSTANCE_OF]},
        {"steps": [wprop.INSTANCE_OF, (wprop.SUBCLASS_OF, 2)]},
        {"steps": [wprop.INSTANCE_OF, wprop.FACET_OF]},
        {"steps": [(wprop.SUBCLASS_OF, 3)]},
        {"steps": [wprop.OCCUPATION], "promote": True},
        {"steps": [wprop.POSITION_HELD], "promote": True},
        {"steps": [wprop.PART_OF, wprop.INSTANCE_OF]},
        {"steps": [wprop.SERIES, wprop.INSTANCE_OF]},
        {"steps": [wprop.SERIES, wprop.LOCATION]},
        {"steps": [wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]},
        {"steps": [wprop.COUNTRY]},
        {"steps": [wprop.CATEGORY_LINK, wprop.CATEGORYS_MAIN_TOPIC]},
        {"steps": [(wprop.CATEGORY_LINK, num_category_link), wprop.FIXED_POINTS]},
        {"steps": [wprop.CATEGORY_LINK, wprop.FIXED_POINTS, wprop.IS_A_LIST_OF]},
        {"steps": [wprop.IS_A_LIST_OF, (wprop.SUBCLASS_OF, 2)]}
    ]
    relation_data = get_relation_data(collection, relations_that_can_extend)
    new_values = values
    # get rid of History of BLAH where link also points to BLAH:

    is_history = IS_HISTORY[new_values]
    is_people_mask = IS_PEOPLE[new_values]
    is_list = IS_LIST_ARTICLE[new_values]
    new_values = related_promote_highest(
        new_values,
        offsets,
        counts,
        condition=is_history,
        alternative=is_people_mask,
        keep_min=keep_min
    )
    unchanged = values == new_values
    is_not_history_or_list = logical_and(
        logical_not(is_history), logical_not(is_list)
    )
    new_values = related_promote_highest(
        new_values,
        offsets,
        counts,
        condition=logical_and(is_history, unchanged),
        alternative=is_not_history_or_list,
        keep_min=keep_min
    )

    is_sport_or_thoroughfare = logical_or(
        IS_EVENT_SPORT, IS_THOROUGHFARE
    )[new_values]

    # delete these references:
    new_values[anchor_length < 2] = -1
    # get rid of shorthand for sports:
    new_values[logical_and(is_sport_or_thoroughfare, anchor_length <= 2)] = -1
    # remove lists of episodes:
    is_episode_list = IS_EPISODE_LIST[new_values]
    new_values[is_episode_list] = -1

    # get rid of "car" -> "Renault Megane", when "car" -> "Car",
    # and "Renault Megane" is instance of "Car":
    is_not_people = logical_not(IS_PEOPLE)[new_values]
    new_values = extend_relations(
        relation_data,
        new_values,
        offsets,
        counts,
        alternative=is_not_people,
        pbar=get_progress_bar("extend_relations", max_value=len(offsets), item="links"),
        keep_min=keep_min
    )
    unchanged = values == new_values
    # remove all non-modified values that are
    # not instances of anything, nor subclasses of anything:
    new_values[logical_ands(
        [
            logical_ands([
                collection.relation(wprop.INSTANCE_OF).edges() == 0,
                collection.relation(wprop.SUBCLASS_OF).edges() == 0,
                collection.relation(wprop.PART_OF).edges() == 0,
                collection.relation(wprop.CATEGORY_LINK).edges() == 0
            ])[new_values],
            unchanged
        ])] = -1

    is_kinship = IS_KINSHIP[new_values]
    is_human = IS_CHARACTER_HUMAN[new_values]
    new_values = related_promote_highest(
        new_values,
        offsets,
        counts,
        condition=is_human,
        alternative=is_kinship,
        keep_min=keep_min
    )

    # replace elements by a country
    # if a better alternative is present,
    # counts is less than 100:
    should_replace_by_country = logical_ands(
        [
            logical_not(
                logical_ors([
                    IS_POLITICAL_ORGANIZATION,
                    IS_CARDINAL_DIRECTION,
                    IS_LANGUAGE_ALPHABET,
                    IS_COUNTRY,
                    IS_PEOPLE_GROUP,
                    IS_BREED,
                    IS_BATTLE,
                    IS_SOCIETY,
                    IS_POSITION,
                    IS_POLITICAL_PARTY,
                    IS_SPORTS_TEAM,
                    IS_CHARACTER_HUMAN,
                    IS_LANDFORM,
                    IS_ACTIVITY
                ])
            )[new_values],
            counts < 100
        ]
    )

    # turn this into a promote highest in this order:
    is_country_or_cardinal = [
        IS_CARDINAL_DIRECTION,
        IS_COUNTRY,
        IS_POLITICAL_ORGANIZATION
    ]
    for i, alternative in enumerate(is_country_or_cardinal):
        unchanged = values == new_values
        should_replace_by_country = logical_and(
            should_replace_by_country, unchanged
        )
        new_values = related_promote_highest(
            new_values,
            offsets,
            counts,
            condition=should_replace_by_country,
            alternative=alternative[new_values],
            keep_min=keep_min
        )

    new_offsets, new_values, new_counts, location_shift = reduce_values(
        offsets, new_values, counts)

    return (new_offsets, new_values, new_counts), location_shift


def filter_trie(trie, values):
    return marisa_trie.Trie((trie.restore_key(value) for value in values))


def remap_trie_offset_array(old_trie, new_trie, offsets_values_counts):
    mapping = np.zeros(len(new_trie), dtype=np.int32)
    t0 = time.time()
    for new_index in range(len(new_trie)):
        mapping[new_index] = old_trie[new_trie.restore_key(new_index)]
    t1 = time.time()
    print("Got mapping from old trie to new trie in %.3fs" % (t1 - t0,))
    ported = []
    for offsets, values, counts in offsets_values_counts:
        new_offsets, new_values, new_counts = remap_offset_array(
            mapping, offsets, values, counts
        )
        ported.append((new_offsets, new_values, new_counts))
    t2 = time.time()
    print("Ported counts and values across tries in %.3fs" % (t2 - t1,))
    return ported


def main():
    args = parse_args()
    if args.new_language_path == args.language_path:
        raise ValueError("new_language_path and language_path must be "
                         "different: cannot generate a fixed trie in "
                         "the same directory as the original trie.")

    c = TypeCollection(args.wikidata, num_names_to_load=0)
    c.load_blacklist(join(SCRIPT_DIR, "blacklist.json"))
    original_values = np.load(
        join(args.language_path, "trie_index2indices_values.npy"))
    original_offsets = np.load(
        join(args.language_path, "trie_index2indices_offsets.npy"))
    original_counts = np.load(
        join(args.language_path, "trie_index2indices_counts.npy"))
    original_trie_path = join(args.language_path, 'trie.marisa')
    trie = marisa_trie.Trie().load(original_trie_path)
    initialize_globals(c)
    t0 = time.time()

    old_location_shift = None
    values, offsets, counts = original_values, original_offsets, original_counts
    for step in range(args.steps):
        anchor_length = get_trie_properties(trie, offsets, values)
        (offsets, values, counts), location_shift = fix(
            collection=c,
            offsets=offsets,
            values=values,
            counts=counts,
            anchor_length=anchor_length,
            num_category_link=8
        )
        if old_location_shift is not None:
            # see where newly shifted values are now pointing
            # to (extra indirection level):
            location_shift = location_shift[old_location_shift]
            location_shift[old_location_shift == -1] = -1
        old_location_shift = location_shift
        pre_reduced_values = values[location_shift]
        pre_reduced_values[location_shift == -1] = -1
        num_changes = int((pre_reduced_values != original_values).sum())
        change_volume = int((original_counts[pre_reduced_values != original_values].sum()))
        print("step %d with %d changes, %d total links" % (
            step, num_changes, change_volume)
        )
    pre_reduced_values = values[location_shift]
    pre_reduced_values[location_shift == -1] = -1
    t1 = time.time()
    num_changes = int((pre_reduced_values != original_values).sum())
    print("Done with link fixing in %.3fs, with %d changes." % (
        t1 - t0, num_changes)
    )

    # show some remappings:
    np.random.seed(1234)
    num_samples = 10
    samples = np.random.choice(
        np.where(
            np.logical_and(
                np.logical_and(
                    pre_reduced_values != original_values,
                    pre_reduced_values != -1
                ),
                original_values != -1
            )
        )[0],
        size=num_samples,
        replace=False
    )
    print("Sample fixes:")
    for index in samples:
        print("   %r (%d) -> %r (%d)" % (
                c.get_name(int(original_values[index])),
                int(original_values[index]),
                c.get_name(int(pre_reduced_values[index])),
                int(pre_reduced_values[index])
            )
        )
    print("")

    samples = np.random.choice(
        np.where(
            OffsetArray(values, offsets).edges() == 0
        )[0],
        size=num_samples,
        replace=False
    )
    print("Sample deletions:")
    for index in samples:
        print("   %r" % (trie.restore_key(int(index))))

    # prune out anchors where there are no more linked items:
    print("Removing empty anchors from trie...")
    t0 = time.time()
    non_empty_offsets = np.where(
        OffsetArray(values, offsets).edges() != 0
    )[0]
    fixed_trie = filter_trie(trie, non_empty_offsets)

    contexts_found = true_exists(
        join(args.language_path, "trie_index2contexts_values.npy")
    )
    if contexts_found:
        contexts_values = np.load(
            join(args.language_path, "trie_index2contexts_values.npy"))
        contexts_offsets = np.load(
            join(args.language_path, "trie_index2contexts_offsets.npy"))
        contexts_counts = np.load(
            join(args.language_path, "trie_index2contexts_counts.npy"))

    to_port = [
        (offsets, values, counts),
        (original_offsets, pre_reduced_values, original_values)
    ]
    if contexts_found:
        to_port.append(
             (contexts_offsets, contexts_values, contexts_counts)
        )

    ported = remap_trie_offset_array(trie, fixed_trie, to_port)
    offsets, values, counts = ported[0]
    original_offsets, pre_reduced_values, original_values = ported[1]
    t1 = time.time()
    print("Removed %d empty anchors from trie in %.3fs" % (
        len(trie) - len(fixed_trie), t1 - t0,)
    )

    print("Saving...")
    makedirs(args.new_language_path, exist_ok=True)

    np.save(join(args.new_language_path, "trie_index2indices_values.npy"),
            values)
    np.save(join(args.new_language_path, "trie_index2indices_offsets.npy"),
            offsets)
    np.save(join(args.new_language_path, "trie_index2indices_counts.npy"),
            counts)
    if contexts_found:
        contexts_offsets, contexts_values, contexts_counts = ported[2]
        np.save(join(args.new_language_path, "trie_index2contexts_values.npy"),
                contexts_values)
        np.save(join(args.new_language_path, "trie_index2contexts_offsets.npy"),
                contexts_offsets)
        np.save(join(args.new_language_path, "trie_index2contexts_counts.npy"),
                contexts_counts)
    new_trie_path = join(args.new_language_path, 'trie.marisa')
    fixed_trie.save(new_trie_path)

    transition = np.vstack([original_values, pre_reduced_values]).T
    np.save(join(args.new_language_path, "trie_index2indices_transition_values.npy"),
            transition)
    np.save(join(args.new_language_path, "trie_index2indices_transition_offsets.npy"),
            original_offsets)
    print("Done.")

if __name__ == "__main__":
    main()
