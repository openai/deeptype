"""
Obtain a coarse-grained classification of places and entities according to their associated
continent/country.
"""
from numpy import (
    logical_and, logical_or, logical_not, logical_xor, where
)
from wikidata_linker_utils.logic import logical_negate
import wikidata_linker_utils.wikidata_properties as wprop


def wkp(c, name):
    """Convert a string wikipedia article name to its Wikidata index."""
    return c.article2id["enwiki/" + name][0][0]


def wkd(c, name):
    """Convert a wikidata QID to its wikidata index."""
    return c.name2index[name]


def classify(c):
    EUROPE = wkp(c, 'Europe')
    AFRICA = wkp(c, 'Africa')
    ASIA = wkp(c, 'Asia')
    NORTH_AMERICA = wkp(c, 'North America')
    SOUTH_AMERICA = wkp(c, 'South America')
    OCEANIA = wkp(c, 'Oceania')
    ANTARCTICA = wkp(c, 'Antarctica')
    CONTINENT = wkp(c, wprop.CONTINENT)
    OUTERSPACE = wkp(c, 'Astronomical object')
    EARTH = wkp(c, "Earth")
    GEOGRAPHIC_LOCATION = wkd(c, "Q2221906")
    POPULATED_PLACE = wkd(c, 'Q486972')

    MIDDLE_EAST = [
        wkp(c, "Bahrain"),
        wkp(c, "Cyprus"),
        wkp(c, "Turkish"),
        wkp(c, "Egypt"),
        wkp(c, "Iran"),
        wkp(c, "Iraq"),
        wkp(c, "Kurdish"),
        wkp(c, "Israel"),
        wkp(c, "Arabic"),
        wkp(c, "Jordan"),
        wkp(c, "Kuwait"),
        wkp(c, "Lebanon"),
        wkp(c, "Oman"),
        wkp(c, "Palestine"),
        wkp(c, "Jordanian"),
        wkp(c, "Qatar"),
        wkp(c, "Saudi Arabia"),
        wkp(c, "Syria"),
        wkp(c, "Turkey"),
        wkp(c, "United Arab Emirates"),
        wkp(c, "Yemen")
    ]


    TRAVERSIBLE = [
        wprop.INSTANCE_OF,
        wprop.SUBCLASS_OF,
        wprop.CONTINENT,
        wprop.PART_OF,
        wprop.COUNTRY_OF_CITIZENSHIP,
        wprop.COUNTRY,
        wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY
    ]
    # c.describe_connection("Q55", "North America", TRAVERSIBLE)
    # return {}
    print("is_in_middle_east")
    is_in_middle_east = c.satisfy(TRAVERSIBLE, MIDDLE_EAST)
    print("is_in_europe")
    is_in_europe = c.satisfy(TRAVERSIBLE, [EUROPE])
    is_in_europe_only = logical_negate(is_in_europe, [is_in_middle_east])
    print("is_in_asia")
    is_in_asia = c.satisfy(TRAVERSIBLE, [ASIA])
    is_in_asia_only = logical_negate(is_in_asia, [is_in_europe, is_in_middle_east])
    print("is_in_africa")
    is_in_africa = c.satisfy(TRAVERSIBLE, [AFRICA])
    is_in_africa_only = logical_negate(is_in_africa, [is_in_europe, is_in_asia, is_in_middle_east])
    print("is_in_north_america")
    is_in_north_america = c.satisfy(TRAVERSIBLE, [NORTH_AMERICA])
    is_in_north_america_only = logical_negate(is_in_north_america, [is_in_europe, is_in_asia, is_in_middle_east])
    print("is_in_south_america")
    is_in_south_america = c.satisfy(TRAVERSIBLE, [SOUTH_AMERICA])
    print("is_in_antarctica")
    is_in_antarctica = c.satisfy(TRAVERSIBLE, [ANTARCTICA])
    is_in_antarctica_only = logical_negate(is_in_antarctica, [is_in_europe, is_in_north_america, is_in_asia, is_in_middle_east])
    print("is_in_oceania")
    is_in_oceania = c.satisfy(TRAVERSIBLE, [OCEANIA])
    is_in_oceania_only = logical_negate(is_in_oceania, [is_in_europe, is_in_north_america, is_in_asia, is_in_middle_east])
    print("is_in_outer_space")
    is_in_outer_space = c.satisfy(TRAVERSIBLE, [OUTERSPACE])
    print("part_of_earth")
    part_of_earth = c.satisfy(
        [wprop.INSTANCE_OF, wprop.PART_OF, wprop.CONTINENT, wprop.COUNTRY_OF_CITIZENSHIP, wprop.COUNTRY, wprop.SUBCLASS_OF],
        [GEOGRAPHIC_LOCATION, EARTH]
    )
    print("is_in_outer_space_not_earth")
    is_in_outer_space_not_earth = logical_negate(
        is_in_outer_space, [part_of_earth]
    )
    print("is_a_populated_place")
    is_populated_place = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [POPULATED_PLACE])
    is_unlocalized_populated_place = logical_negate( is_populated_place, [is_in_europe, is_in_asia, is_in_antarctica, is_in_oceania, is_in_outer_space, is_in_south_america, is_in_north_america])

    return {
        "europe": is_in_europe_only,
        "asia": is_in_asia_only,
        "africa": is_in_africa_only,
        "middle_east": is_in_middle_east,
        "north_america": is_in_north_america_only,
        "south_america": is_in_south_america,
        "antarctica": is_in_antarctica_only,
        "oceania": is_in_oceania_only,
        "outer_space": is_in_outer_space_not_earth,
        # "populated_space": is_populated_place,
        "populated_place_unlocalized": is_unlocalized_populated_place
    }
