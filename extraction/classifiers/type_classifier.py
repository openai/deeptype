"""
Associate to each entity a type (exclusive membership). Association is imperfect
(e.g. some false positives, false negatives), however the majority of entities
are covered under this umbrella and thus a model can learn to predict several
of the attributes listed below.
"""
from numpy import (
    logical_and, logical_or, logical_not, logical_xor, where
)
from wikidata_linker_utils.logic import logical_negate, logical_ors, logical_ands
import wikidata_linker_utils.wikidata_properties as wprop

def wkp(c, name):
    return c.article2id['enwiki/' + name][0][0]

def wkd(c, name):
    return c.name2index[name]


def classify(c):
    TRAVERSIBLE = [wprop.INSTANCE_OF, wprop.SUBCLASS_OF]
    TRAVERSIBLE_LO = [wprop.INSTANCE_OF, wprop.SUBCLASS_OF, wprop.IS_A_LIST_OF]

    MALE = wkd(c,"Q6581097")
    FEMALE = wkd(c,"Q6581072")
    HUMAN = wkp(c, "Human")
    TAXON = wkd(c, "Q16521")
    HORSE = wkd(c, "Q726")
    RACE_HORSE = wkd(c, "Q10855242")
    FOSSIL_TAXON = wkd(c, "Q23038290")
    MONOTYPIC_TAXON = wkd(c, "Q310890")
    FOOD = wkp(c, "Food")
    DRINK = wkp(c, "Drink")
    BIOLOGY = wkp(c, "Biology")
    GEOGRAPHICAL_OBJECT = wkd(c, "Q618123")
    LOCATION_GEOGRAPHY = wkd(c, "Q2221906")
    ORGANISATION = wkp(c, 'Organization')
    MUSICAL_WORK = wkd(c, 'Q2188189')
    AUDIO_VISUAL_WORK = wkd(c,'Q2431196')
    ART_WORK = wkd(c,'Q838948')
    PHYSICAL_OBJECT = wkp(c, "Physical body")
    VALUE = wkd(c, 'Q614112')
    TIME_INTERVAL = wkd(c, 'Q186081')
    EVENT = wkd(c, 'Q1656682')
    POPULATED_PLACE = wkd(c, 'Q486972')
    ACTIVITY = wkd(c, "Q1914636")
    PROCESS = wkd(c, "Q3249551")
    BODY_OF_WATER = wkd(c, "Q15324")
    PEOPLE = wkd(c, "Q2472587")
    LANGUAGE = wkd(c, "Q34770")
    ALPHABET = wkd(c, "Q9779")
    SPEECH = wkd(c, "Q861911")
    GAS = wkd(c, "Q11432")
    CHEMICAL_COMPOUND = wkd(c, "Q11173")
    DRUG = wkd(c, "Q8386")
    GEOMETRIC_SHAPE = wkd(c, "Q815741")
    MIND = wkd(c, "Q450")
    TV_STATION = wkd(c, "Q1616075")

    AWARD_CEREMONY = wkd(c, "Q4504495")
    SONG = wkd(c, "Q7366")
    SINGLE = wkd(c, "Q134556")
    CHESS_OPENING = wkd(c, "Q103632")
    BATTLE = wkd(c, "Q178561")
    BLOCKADE = wkd(c, "Q273976")
    MILITARY_OFFENSIVE = wkd(c, "Q2001676")
    DEVELOPMENT_BIOLOGY = wkd(c, "Q213713")
    UNIT_OF_MASS = wkd(c, "Q3647172")
    WATERCOURSE = wkd(c, "Q355304")
    VOLCANO = wkd(c, "Q8072")
    LAKE = wkd(c, "Q23397")
    SEA = wkd(c, "Q165")
    BRAND = wkd(c, "Q431289")
    AUTOMOBILE_MANUFACTURER = wkd(c, "Q786820")
    MOUNTAIN = wkd(c, "Q8502")
    MASSIF = wkd(c, "Q1061151")
    WAR = wkd(c, "Q198")
    CRIME = wkd(c, "Q83267")
    GENE = wkd(c, "Q7187")
    CHROMOSOME = wkd(c, "Q37748")
    DISEASE = wkd(c, "Q12136")
    ASTEROID = wkd(c, "Q3863")
    COMET = wkd(c, "Q3559")
    PLANET = wkd(c, "Q634")
    GALAXY = wkd(c, "Q318")
    ASTRONOMICAL_OBJECT = wkd(c, "Q6999")
    FICTIONAL_ASTRONOMICAL_OBJECT = wkd(c, "Q15831598")
    MATHEMATICAL_OBJECT = wkd(c, "Q246672")
    REGION = wkd(c, "Q82794")
    PHYSICAL_QUANTITY = wkd(c, "Q107715")
    NUMBER = wkd(c, "Q11563")
    NATURAL_PHENOMENON = wkd(c, "Q1322005")
    GEOLOGICAL_FORMATION = wkd(c, "Q736917")
    CURRENCY = wkd(c, "Q8142")
    MONEY = wkd(c, "Q1368")
    LANDFORM = wkd(c, "Q271669")
    COUNTRY = wkd(c, "Q6256")
    FICTIONAL_HUMAN = wkd(c, "Q15632617")
    AWARD = wkd(c, "Q618779")
    RELIGIOUS_TEXT = wkd(c, "Q179461")
    OCCUPATION = wkd(c, "Q12737077")
    PROFESSION = wkd(c, "Q28640")
    POSITION = wkd(c, "Q4164871")
    RELIGION = wkd(c, "Q9174")
    SOFTWARE = wkd(c, "Q7397")
    ELECTRONIC_GAME = wkd(c, "Q2249149")
    GAME = wkd(c, "Q11410")
    VIDEO_GAME_FRANCHISES = wkd(c, "Q7213857")
    TRAIN_STATION = wkd(c, "Q55488")
    BRIDGE = wkd(c, "Q12280")
    AIRPORT = wkd(c, "Q62447")
    SURNAME = wkd(c, "Q101352")
    GIVEN_NAME = wkd(c, "Q202444")
    FEMALE_GIVEN_NAME = wkd(c, "Q11879590")
    MALE_GIVEN_NAME = wkd(c, "Q12308941")
    GIVEN_NAME = wkd(c, "Q202444")
    MOLECULE = wkd(c, "Q11369")
    PROTEIN_FAMILY = wkd(c, "Q417841")
    PROTEIN_DOMAIN = wkd(c, "Q898273")
    MULTIPROTEIN_COMPLEX = wkd(c, "Q420927")
    LAW = wkd(c, "Q7748")
    VEHICLE = wkd(c, "Q42889")
    MODE_OF_TRANSPORT = wkd(c, "Q334166")
    WATERCRAFT = wkd(c, "Q1229765")
    AIRCRAFT = wkd(c, "Q11436")
    ROAD_VEHICLE = wkd(c, "Q1515493")
    AUTOMOBILE_MODEL = wkd(c, "Q3231690")
    AUTOMOBILE = wkd(c, "Q1420")
    TRUCK = wkd(c, "Q43193")
    MOTORCYCLE_MODEL = wkd(c, "Q23866334")
    TANK = wkd(c, "Q12876")
    FIRE_ENGINE = wkd(c, "Q208281")
    AMBULANCE = wkd(c, "Q180481")
    RAILROAD = wkd(c, "Q22667")
    RADIO_PROGRAM = wkd(c, "Q1555508")
    DISCOGRAPHY = wkd(c, "Q273057")
    WEBSITE = wkd(c, "Q35127")
    WEAPON = wkd(c, "Q728")
    PUBLICATION = wkd(c, "Q732577")
    ARTICLE = wkd(c, "Q191067")
    FAMILY = wkd(c, "Q8436")
    FICTIONAL_CHARACTER = wkd(c, "Q95074")
    FACILITY = wkd(c, "Q13226383")
    CONCEPT = wkd(c, "Q151885")
    PROVERB = wkd(c, "Q35102")
    ANATOMICAL_STRUCTURE = wkd(c, "Q4936952")
    BREED = wkd(c, "Q38829")
    PLANT_STRUCTURE = wkd(c, "Q25571752")
    PLANT = wkd(c, "Q756")
    SPECIAL_FIELD = wkd(c, "Q1047113")
    ACADEMIC_DISCIPLINE = wkd(c, "Q11862829")
    TERM = wkd(c, "Q1969448")
    SEXUAL_ORIENTATION = wkd(c, "Q17888")
    PARADIGM = wkd(c, "Q28643")
    LEGAL_CASE = wkd(c, "Q2334719")
    SPORT = wkd(c, "Q349")
    RECURRING_SPORTING_EVENT = wkd(c, "Q18608583")
    ART_GENRE = wkd(c, "Q1792379")
    SPORTING_EVENT = wkd(c, "Q16510064")
    COMIC = wkd(c, "Q1004")
    CHARACTER = wkd(c, "Q3241972")
    PERSON = wkd(c, "Q215627")
    NATIONAL_HERITAGE_SITE = wkd(c, "Q358")
    ESTATE = wkd(c, "Q2186896")
    ELECTION = wkd(c, "Q40231")
    LEGISLATIVE_TERM = wkd(c, "Q15238777")
    COMPETITION = wkd(c, "Q476300")
    LEGAL_ACTION = wkd(c, "Q27095657")
    SEX_TOY = wkd(c, "Q10816")
    MONUMENT = wkd(c, "Q4989906")
    ASSOCIATION_FOOTBALL_POSITION = wkd(c, "Q4611891")
    # ICE_HOCKEY_POSITION = wkd(c, "Q18533987")
    # PART_OF_LAND = wkd(c, "Q23001306")
    MUSIC_DOWNLOAD = wkd(c, "Q6473564")
    OCCUPATION = wkd(c, "Q12737077")
    KINSHIP = wkd(c, "Q171318")
    KIN = wkd(c, "Q21073947")
    PSEUDONYM = wkd(c, "Q61002")
    STOCK_CHARACTER = wkd(c, "Q162244")
    TITLE = wkd(c, "Q4189293")
    DATA_FORMAT = wkd(c, "Q494823")
    ELECTROMAGNETIC_WAVE = wkd(c, "Q11386")
    POSTAL_CODE = wkd(c, "Q37447")
    CLOTHING = wkd(c, "Q11460")
    NATIONALITY = wkd(c, "Q231002")
    BASEBALL_POSITION = wkd(c, "Q1151733")
    AMERICAN_FOOTBALL_POSITIONS = wkd(c, "Q694589")
    POSITION_TEAM_SPORTS = wkd(c, "Q1781513")
    FILE_FORMAT_FAMILY = wkd(c, "Q26085352")
    FILE_FORMAT = wkd(c, "Q235557")
    TAXONOMIC_RANK = wkd(c, "Q427626")
    ORDER_HONOUR = wkd(c, "Q193622")
    BRANCH_OF_SCIENCE = wkd(c, "Q2465832")
    RESEARCH = wkd(c, "Q42240")
    METHOD = wkd(c, "Q1799072")
    ALGORITHM = wkd(c, "Q8366")
    PROPOSITION = wkd(c, "Q108163")
    SPORTSPERSON = wkd(c, "Q2066131")
    LAKES_MINESOTTA = wkd(c, "Q8580663")
    NAMED_PASSENGER_TRAIN_INDIA = wkd(c, "Q9260591")
    TOWNSHIPS_MISOURI = wkd(c, "Q8861637")
    RACE_ETHNICITY_USA = wkd(c, "Q2035701")
    RECORD_CHART = wkd(c, "Q373899")
    SINGLE_ENGINE_AIRCRAFT = wkd(c, "Q7405339")
    SIGNIFICANT_OTHER = wkd(c, "Q841509")
    BILLBOARDS = wkd(c, "Q19754079")
    RADIO_STATION = wkd(c, "Q19754079")
    RADIO_STATION2 = wkd(c, "Q1474493")
    NOBLE_TITLE = wkd(c, "Q216353")
    HOUSES_NATIONAL_REGISTER_ARKANSAS = wkd(c, "Q8526394")
    CLADE = wkd(c, "Q713623")
    BOARD_GAMES = wkd(c, "Q131436")
    CLAN = wkd(c, "Q211503")
    ACCIDENT = wkd(c, "Q171558")
    MASSACRE = wkd(c, "Q3199915")
    TORNADO = wkd(c, "Q8081")
    NATURAL_DISASTER = wkd(c, "Q8065")
    SPORTS_TEAM = wkd(c, "Q12973014")
    BAND_ROCK_AND_POP = wkd(c, "Q215380")
    ORGANIZATION_OTHER = wkd(c, "Q43229")
    POLITICAL_PARTY = wkd(c, "Q7278")
    SPECIES = wkd(c, "Q7432")
    CHEMICAL_SUBSTANCE = wkd(c, "Q79529")

    THREATENED_SPECIES = wkd(c, "Q515487")
    HYPOTHETICAL_SPECIES = wkd(c, "Q5961273")

    CONFLICT = wkd(c, "Q180684")
    PRIVATE_USE_AREAS = wkd(c, "Q11152836")

    BARONETCIES_IN_UK = wkd(c, "Q8290061")
    EXTINCT_BARONETCIES_ENGLAND = wkd(c, "Q8432223")
    EXTINCT_BARONETCIES_UK = wkd(c, "Q8432226")

    WIKIPEDIA_DISAMBIGUATION = wkd(c, "Q4167410")
    WIKIPEDIA_TEMPLATE_NAMESPACE = wkd(c, "Q11266439")
    WIKIPEDIA_LIST = wkd(c, "Q13406463")
    WIKIPEDIA_PROJECT_PAGE = wkd(c, "Q14204246")
    WIKIMEDIA_CATEGORY_PAGE = wkd(c, "Q4167836")
    WIKIPEDIA_USER_LANGUAGE_TEMPLATE = wkd(c, "Q19842659")
    WIKIDATA_PROPERTY = wkd(c, "Q18616576")
    COLLEGIATE_ATHLETICS_PROGRAM = wkd(c, "Q5146583")
    SPORTS_TRANSFER_AF = wkd(c, "Q1811518")
    DEMOGRAPHICS_OF_NORWAY = wkd(c, "Q7664203")
    DOCUMENT = wkd(c, "Q49848")
    BASIC_STAT_UNIT_NORWAY = wkd(c, "Q4580177")
    PUBLIC_TRANSPORT = wkd(c, "Q178512")
    HAZARD = wkd(c, "Q1132455")
    BASEBALL_RULES = wkd(c, "Q1153773")
    HIT_BASEBALL = wkd(c, "Q713493")
    OUT_BASEBALL = wkd(c, "Q1153773")
    LAWS_OF_ASSOCIATION_FOOTBALL = wkd(c, "Q7215850")
    CRICKET_LAWS_AND_REGULATION = wkd(c, "Q8427034")
    MEASUREMENTS_OF_POVERTY = wkd(c, "Q8614855")
    PROFESSIONAL_WRESTLING_MATCH_TYPES = wkd(c, "Q679633")
    CITATION = wkd(c, "Q1713")
    INTERNATIONAL_RELATIONS = wkd(c, "Q166542")
    WORLD_VIEW = wkd(c, "Q49447")
    ROCK_GEOLOGY = wkd(c, "Q8063")
    BASEBALL_STATISTIC = wkd(c, "Q8291081")
    BASEBALL_STATISTICS = wkd(c, "Q809898")
    TRAIN_ACCIDENT = wkd(c, "Q1078765")
    CIRCUS_SKILLS = wkd(c, "Q4990963")
    FOLKLORE = wkd(c, "Q36192")
    NEWS_BUREAU = wkd(c, "Q19824398")
    RECESSION = wkd(c, "Q176494")
    NYC_BALLET = wkd(c, "Q1336942")
    SPORTS_RECORD = wkd(c, "Q1241356")
    WINGSPAN = wkd(c, "Q245097")
    WIN_LOSS_RECORD_PITCHING = wkd(c, "Q1202506")
    CRICKET_TERMINOLOGY = wkd(c, "Q8427141")
    UNION_ARMY = wkd(c, "Q1752901")
    POPULATION = wkd(c, "Q33829")
    WIND = wkd(c, "Q8094")
    TORPEDO_TUBE = wkd(c, "Q1330003")
    WEAPONS_PLATFORM = wkd(c, "Q7978115")
    COLOR = wkd(c, "Q1075")
    SOCIAL_SCIENCE = wkd(c, "Q34749")
    DISCIPLINE_ACADEMIA = wkd(c, "Q11862829")
    FORMAL_SCIENCE = wkd(c, "Q816264")
    ASPHALT = wkd(c, "Q167510")
    TALK_RADIO = wkd(c, "Q502319")
    ART_MOVEMENT = wkd(c, "Q968159")
    IDEOLOGY = wkd(c, "Q7257")

    # print([c.get_name(idx) for idx in c.relation(wprop.INSTANCE_OF)[wkd(c, "Q14934048")]])
    # print([c.get_name(idx) for idx in c.get_inverted_relation(wprop.INSTANCE_OF)[wkd(c, "Q14934048")]])

    # print([c.get_name(idx) for idx in c.relation(wprop.PART_OF)[wkd(c, "Q14934048")]])
    # print([c.get_name(idx) for idx in c.get_inverted_relation(wprop.PART_OF)[wkd(c, "Q14934048")]])

    # print([c.get_name(idx) for idx in c.relation(wprop.SUBCLASS_OF)[wkd(c, "Q14934048")]])
    # print([c.get_name(idx) for idx in c.get_inverted_relation(wprop.SUBCLASS_OF)[wkd(c, "Q14934048")]])

    # print([c.get_name(idx) for idx in c.relation(wprop.CATEGORY_LINK)[wkd(c, "Q14934048")]])
    # print([c.get_name(idx) for idx in c.get_inverted_relation(wprop.CATEGORY_LINK)[wkd(c, "Q14934048")]])

    is_sports_terminology = logical_or(
        c.satisfy(TRAVERSIBLE_LO, [OUT_BASEBALL, HIT_BASEBALL]),
        c.satisfy(
            [wprop.CATEGORY_LINK],
            [
                BASEBALL_RULES,
                LAWS_OF_ASSOCIATION_FOOTBALL,
                CRICKET_LAWS_AND_REGULATION,
                PROFESSIONAL_WRESTLING_MATCH_TYPES,
                CRICKET_TERMINOLOGY
            ],
            max_steps=1
        )
    )
    is_accident = c.satisfy(TRAVERSIBLE_LO, [ACCIDENT])
    is_taxon = c.satisfy([wprop.INSTANCE_OF, wprop.IS_A_LIST_OF],
        [
            TAXON, FOSSIL_TAXON, MONOTYPIC_TAXON, HORSE, RACE_HORSE, CLADE, SPECIES,
            THREATENED_SPECIES, HYPOTHETICAL_SPECIES
        ]
    )
    is_breed = c.satisfy(TRAVERSIBLE_LO, [BREED])
    is_taxon_or_breed = logical_or(is_taxon, is_breed)
    is_human = c.satisfy(TRAVERSIBLE_LO, [HUMAN, FICTIONAL_HUMAN])
    is_country = c.satisfy(TRAVERSIBLE_LO, [COUNTRY])
    is_people = c.satisfy(
        TRAVERSIBLE_LO,
        [
            PEOPLE,
            NATIONALITY,
            SPORTS_TRANSFER_AF,
            POPULATION
        ]
    )
    is_populated_place = logical_or(
        c.satisfy(TRAVERSIBLE_LO, [POPULATED_PLACE]),
        c.satisfy([wprop.CATEGORY_LINK], [TOWNSHIPS_MISOURI], max_steps=1)
    )
    is_organization = c.satisfy(
        TRAVERSIBLE_LO,
        [
            POLITICAL_PARTY,
            COLLEGIATE_ATHLETICS_PROGRAM,
            ORGANIZATION_OTHER,
            ORGANISATION,
            SPORTS_TEAM,
            BAND_ROCK_AND_POP,
            NEWS_BUREAU,
            NYC_BALLET,
            UNION_ARMY
        ]
    )
    is_position = c.satisfy(
        TRAVERSIBLE_LO,
        [
            POSITION,
            OCCUPATION,
            POSITION_TEAM_SPORTS,
            AMERICAN_FOOTBALL_POSITIONS,
            ASSOCIATION_FOOTBALL_POSITION,
            BASEBALL_POSITION,
            # ICE_HOCKEY_POSITION,
            SPORTSPERSON
        ]
    )
    is_kinship = c.satisfy(TRAVERSIBLE_LO, [KINSHIP])
    is_kin = c.satisfy([wprop.SUBCLASS_OF, wprop.IS_A_LIST_OF], [KIN])
    is_title = logical_or(
        c.satisfy(TRAVERSIBLE_LO, [TITLE, NOBLE_TITLE]),
        c.satisfy([wprop.CATEGORY_LINK], [BARONETCIES_IN_UK, EXTINCT_BARONETCIES_UK, EXTINCT_BARONETCIES_ENGLAND], max_steps=1)
    )
    is_art_work = c.satisfy(TRAVERSIBLE_LO, [ART_WORK, COMIC])
    is_audio_visual_work = c.satisfy(TRAVERSIBLE_LO, [AUDIO_VISUAL_WORK, TV_STATION])
    is_fictional_character = c.satisfy(TRAVERSIBLE_LO, [FICTIONAL_CHARACTER])
    is_name = c.satisfy(TRAVERSIBLE_LO, [GIVEN_NAME, SURNAME, FEMALE_GIVEN_NAME, MALE_GIVEN_NAME, PSEUDONYM])
    is_stock_character = c.satisfy([wprop.INSTANCE_OF, wprop.IS_A_LIST_OF], [STOCK_CHARACTER])
    is_family = c.satisfy(TRAVERSIBLE_LO, [FAMILY, CLAN])
    is_award = c.satisfy(TRAVERSIBLE_LO, [AWARD])
    is_electromagnetic_wave = c.satisfy(TRAVERSIBLE_LO, [ELECTROMAGNETIC_WAVE])
    is_geographical_object = c.satisfy(
        TRAVERSIBLE_LO,
        [
            GEOGRAPHICAL_OBJECT,
            BODY_OF_WATER,
            LOCATION_GEOGRAPHY,
            GEOLOGICAL_FORMATION,
            NATIONAL_HERITAGE_SITE,
            ESTATE,
            # PART_OF_LAND,
            PRIVATE_USE_AREAS
        ]
    )
    is_postal_code = c.satisfy(TRAVERSIBLE_LO, [POSTAL_CODE])
    is_person = c.satisfy(TRAVERSIBLE_LO, [PERSON])
    is_person_only = logical_or(
        logical_negate(
        is_person,
        [
            is_human,
            is_people,
            is_populated_place,
            is_organization,
            is_position,
            is_title,
            is_kinship,
            is_kin,
            is_country,
            is_geographical_object,
            is_art_work,
            is_audio_visual_work,
            is_fictional_character,
            is_name,
            is_family,
            is_award
        ]
    ), is_stock_character)

    is_male = c.satisfy([wprop.SEX_OR_GENDER], [MALE])
    is_female = c.satisfy([wprop.SEX_OR_GENDER], [FEMALE])
    is_human_male = logical_and(is_human, is_male)
    is_human_female = logical_and(is_human, is_female)

    is_musical_work = c.satisfy(TRAVERSIBLE_LO, [MUSICAL_WORK, DISCOGRAPHY])
    is_song = c.satisfy(TRAVERSIBLE_LO, [SONG, SINGLE])
    is_radio_program = c.satisfy(
        TRAVERSIBLE_LO,
        [
            RADIO_PROGRAM,
            RADIO_STATION,
            RADIO_STATION2,
            TALK_RADIO
        ]
    )
    is_sexual_orientation = c.satisfy(TRAVERSIBLE_LO, [SEXUAL_ORIENTATION])

    is_taxonomic_rank = c.satisfy([wprop.INSTANCE_OF], [TAXONOMIC_RANK])
    is_order = c.satisfy(TRAVERSIBLE_LO, [ORDER_HONOUR])

    is_train_station = c.satisfy(TRAVERSIBLE_LO, [TRAIN_STATION])
    is_bridge = c.satisfy(TRAVERSIBLE_LO, [BRIDGE])
    is_airport = c.satisfy(TRAVERSIBLE_LO, [AIRPORT])

    is_sex_toy = c.satisfy(TRAVERSIBLE_LO, [SEX_TOY])
    is_monument = c.satisfy(TRAVERSIBLE_LO, [MONUMENT])

    is_physical_object = c.satisfy(
        TRAVERSIBLE_LO,
        [
            PHYSICAL_OBJECT,
            BOARD_GAMES,
            ELECTRONIC_GAME,
            GAME,
            ROCK_GEOLOGY,
            ASPHALT
        ]
    )
    is_clothing = c.satisfy(TRAVERSIBLE_LO, [CLOTHING])

    is_mathematical_object = c.satisfy(TRAVERSIBLE_LO, [MATHEMATICAL_OBJECT])
    is_physical_quantity = logical_or(
        c.satisfy(
            TRAVERSIBLE_LO,
            [
                PHYSICAL_QUANTITY,
                BASIC_STAT_UNIT_NORWAY,
                SPORTS_RECORD,
                WINGSPAN,
                WIN_LOSS_RECORD_PITCHING,
                BASEBALL_STATISTICS
            ]
        ),
        c.satisfy(
            [wprop.CATEGORY_LINK],
            [
                DEMOGRAPHICS_OF_NORWAY,
                MEASUREMENTS_OF_POVERTY,
                BASEBALL_STATISTIC
            ],
            max_steps=1
        )
    )
    is_number = c.satisfy(TRAVERSIBLE_LO, [NUMBER])
    is_astronomical_object = c.satisfy(
        TRAVERSIBLE_LO,
        [
            ASTEROID,
            COMET,
            PLANET,
            GALAXY,
            ASTRONOMICAL_OBJECT,
            FICTIONAL_ASTRONOMICAL_OBJECT
        ]
    )
    is_hazard = c.satisfy(TRAVERSIBLE_LO, [HAZARD, TRAIN_ACCIDENT])
    is_date = c.satisfy(TRAVERSIBLE_LO, [TIME_INTERVAL])
    is_algorithm = c.satisfy(TRAVERSIBLE_LO, [ALGORITHM])
    is_value = c.satisfy(TRAVERSIBLE_LO, [VALUE])
    is_currency = c.satisfy(TRAVERSIBLE_LO, [CURRENCY, MONEY])
    is_event = c.satisfy(TRAVERSIBLE_LO, [EVENT, RECESSION])
    is_election = c.satisfy(TRAVERSIBLE_LO, [ELECTION])
    is_legislative_term = c.satisfy(TRAVERSIBLE_LO, [LEGISLATIVE_TERM])
    is_activity = c.satisfy([wprop.INSTANCE_OF, wprop.IS_A_LIST_OF], [ACTIVITY, MUSIC_DOWNLOAD, CIRCUS_SKILLS])
    is_activity_subclass = c.satisfy([wprop.SUBCLASS_OF], [ACTIVITY, MUSIC_DOWNLOAD, CIRCUS_SKILLS])
    is_food = c.satisfy([wprop.INSTANCE_OF, wprop.PART_OF, wprop.SUBCLASS_OF], [FOOD, DRINK])
    is_wikidata_prop = c.satisfy(TRAVERSIBLE_LO, [WIKIDATA_PROPERTY])
    is_wikipedia_disambiguation = c.satisfy([wprop.INSTANCE_OF], [WIKIPEDIA_DISAMBIGUATION])
    is_wikipedia_template_namespace = c.satisfy([wprop.INSTANCE_OF], [WIKIPEDIA_TEMPLATE_NAMESPACE])
    is_wikipedia_list = c.satisfy([wprop.INSTANCE_OF], [WIKIPEDIA_LIST])
    is_wikipedia_project_page = c.satisfy([wprop.INSTANCE_OF], [WIKIPEDIA_PROJECT_PAGE])
    is_wikipedia_user_language_template = c.satisfy([wprop.INSTANCE_OF], [WIKIPEDIA_USER_LANGUAGE_TEMPLATE])
    is_wikimedia_category_page = c.satisfy([wprop.INSTANCE_OF], [WIKIMEDIA_CATEGORY_PAGE])
    is_legal_case = c.satisfy(TRAVERSIBLE_LO, [LEGAL_CASE])
    is_sport = c.satisfy(TRAVERSIBLE_LO, [SPORT])
    is_data_format = c.satisfy(TRAVERSIBLE_LO, [DATA_FORMAT, FILE_FORMAT_FAMILY, FILE_FORMAT])
    is_research_method = c.satisfy(TRAVERSIBLE_LO, [RESEARCH, METHOD, RACE_ETHNICITY_USA])
    is_proposition = c.satisfy(TRAVERSIBLE_LO, [PROPOSITION])
    is_record_chart = c.satisfy(TRAVERSIBLE_LO, [RECORD_CHART, BILLBOARDS])
    is_international_relations = c.satisfy(TRAVERSIBLE_LO, [INTERNATIONAL_RELATIONS])

    is_union = c.satisfy(TRAVERSIBLE_LO, [SIGNIFICANT_OTHER])

    is_recurring_sporting_event = c.satisfy(
        TRAVERSIBLE_LO,
        [RECURRING_SPORTING_EVENT]
    )
    is_sport_event = logical_or(
        logical_and(
            is_sport,
            c.satisfy([wprop.PART_OF, wprop.IS_A_LIST_OF], where(is_recurring_sporting_event)[0])
        ),
        c.satisfy(TRAVERSIBLE_LO, [SPORTING_EVENT, COMPETITION])
    )

    is_genre = c.satisfy(TRAVERSIBLE_LO, [ART_GENRE, ART_MOVEMENT])

    is_landform = c.satisfy(TRAVERSIBLE_LO, [LANDFORM])
    is_language = c.satisfy(TRAVERSIBLE_LO, [LANGUAGE])
    is_alphabet = c.satisfy(TRAVERSIBLE_LO, [ALPHABET])
    is_railroad = logical_or(
        c.satisfy(TRAVERSIBLE_LO, [RAILROAD]),
        c.satisfy([wprop.CATEGORY_LINK], [NAMED_PASSENGER_TRAIN_INDIA], max_steps=1)
    )
    is_speech = c.satisfy(TRAVERSIBLE_LO, [SPEECH])
    is_language_only = logical_negate(is_language, [is_speech])
    is_alphabet_only = logical_negate(is_alphabet, [is_speech, is_language])
    is_war = c.satisfy(TRAVERSIBLE_LO, [WAR])
    is_battle = c.satisfy(TRAVERSIBLE_LO, [BATTLE, BLOCKADE, MILITARY_OFFENSIVE, CONFLICT, MASSACRE])
    is_crime = c.satisfy(TRAVERSIBLE_LO, [CRIME])
    is_gas = c.satisfy(TRAVERSIBLE_LO, [GAS])
    is_chemical_compound = c.satisfy(TRAVERSIBLE_LO, [CHEMICAL_COMPOUND, DRUG, CHEMICAL_SUBSTANCE])
    is_chemical_compound_only = logical_negate(is_chemical_compound, [is_food])
    is_gas_only = logical_negate(is_gas, [is_chemical_compound])
    is_geometric_shape = c.satisfy(TRAVERSIBLE_LO, [GEOMETRIC_SHAPE])
    is_award_ceremony = c.satisfy(TRAVERSIBLE_LO, [AWARD_CEREMONY])
    is_strategy = c.satisfy(TRAVERSIBLE_LO, [CHESS_OPENING])
    is_gene = c.satisfy(TRAVERSIBLE_LO, [GENE, CHROMOSOME])
    is_character = c.satisfy(TRAVERSIBLE_LO, [CHARACTER])
    is_law = c.satisfy(TRAVERSIBLE_LO, [LAW])
    is_legal_action = c.satisfy(TRAVERSIBLE_LO, [LEGAL_ACTION])
    is_facility = logical_or(
        c.satisfy(TRAVERSIBLE_LO, [FACILITY]),
        c.satisfy([wprop.CATEGORY_LINK], [HOUSES_NATIONAL_REGISTER_ARKANSAS], max_steps=1)
    )
    is_molecule = c.satisfy(TRAVERSIBLE_LO, [MOLECULE, PROTEIN_FAMILY, PROTEIN_DOMAIN, MULTIPROTEIN_COMPLEX])
    is_disease = c.satisfy(TRAVERSIBLE_LO, [DISEASE])
    is_mind = c.satisfy(TRAVERSIBLE_LO, [MIND])
    is_religion = c.satisfy(TRAVERSIBLE_LO, [RELIGION])
    is_natural_phenomenon = c.satisfy(TRAVERSIBLE_LO, [NATURAL_PHENOMENON, NATURAL_DISASTER, WIND])
    is_anatomical_structure = c.satisfy(TRAVERSIBLE_LO, [ANATOMICAL_STRUCTURE])
    is_plant = c.satisfy(TRAVERSIBLE_LO + [wprop.PARENT_TAXON], [PLANT_STRUCTURE, PLANT])
    is_region = c.satisfy(TRAVERSIBLE_LO, [REGION])
    is_software = logical_or(
        c.satisfy(TRAVERSIBLE_LO, [SOFTWARE]),
        c.satisfy([wprop.CATEGORY_LINK], [VIDEO_GAME_FRANCHISES], max_steps=1)
    )
    is_website = c.satisfy(TRAVERSIBLE_LO, [WEBSITE])
    is_river = logical_and(c.satisfy(TRAVERSIBLE_LO, [WATERCOURSE]), is_geographical_object)
    is_lake = logical_or(
        logical_and(c.satisfy(TRAVERSIBLE_LO, [LAKE]), is_geographical_object),
        c.satisfy([wprop.CATEGORY_LINK], [LAKES_MINESOTTA], max_steps=1)
    )
    is_sea = logical_and(c.satisfy(TRAVERSIBLE_LO, [SEA]), is_geographical_object)
    is_volcano = logical_and(c.satisfy(TRAVERSIBLE_LO, [VOLCANO]), is_geographical_object)

    is_development_biology = c.satisfy([wprop.PART_OF, wprop.SUBCLASS_OF, wprop.INSTANCE_OF], [DEVELOPMENT_BIOLOGY, BIOLOGY])
    is_unit_of_mass = c.satisfy(TRAVERSIBLE_LO, [UNIT_OF_MASS])
    is_vehicle = c.satisfy(TRAVERSIBLE_LO, [VEHICLE, MODE_OF_TRANSPORT, PUBLIC_TRANSPORT])
    is_watercraft = c.satisfy(TRAVERSIBLE_LO, [WATERCRAFT])
    is_aircraft = logical_or(
        c.satisfy(TRAVERSIBLE_LO, [AIRCRAFT]),
        c.satisfy([wprop.CATEGORY_LINK], [SINGLE_ENGINE_AIRCRAFT], max_steps=1)
    )
    is_road_vehicle = c.satisfy(
        TRAVERSIBLE_LO,
        [
            ROAD_VEHICLE,
            TANK,
            FIRE_ENGINE,
            AMBULANCE,
            AUTOMOBILE_MODEL,
            MOTORCYCLE_MODEL
        ]
    )
    is_weapon = c.satisfy(TRAVERSIBLE_LO, [WEAPON, TORPEDO_TUBE, WEAPONS_PLATFORM])
    is_book_magazine_article_proverb = c.satisfy(
        TRAVERSIBLE_LO,
        [
            PUBLICATION,
            ARTICLE,
            RELIGIOUS_TEXT,
            PROVERB,
            DOCUMENT,
            CITATION,
            FOLKLORE
        ]
    )
    is_brand = c.satisfy(TRAVERSIBLE_LO, [BRAND])
    is_concept = logical_or(
        c.satisfy([wprop.INSTANCE_OF],
            [TERM, ACADEMIC_DISCIPLINE, SPECIAL_FIELD, BRANCH_OF_SCIENCE, WORLD_VIEW]
        ),
        c.satisfy([wprop.SUBCLASS_OF], [SOCIAL_SCIENCE, DISCIPLINE_ACADEMIA, FORMAL_SCIENCE, IDEOLOGY])
    )
    is_color = c.satisfy(TRAVERSIBLE_LO, [COLOR])
    is_paradigm = c.satisfy(TRAVERSIBLE_LO, [PARADIGM])
    is_vehicle_brand = logical_or(
        logical_and(c.satisfy([wprop.PRODUCT_OR_MATERIAL_PRODUCED], [AUTOMOBILE, TRUCK]), is_brand),
        c.satisfy(TRAVERSIBLE_LO, [AUTOMOBILE_MANUFACTURER])
    )
    is_mountain_massif = logical_and(c.satisfy(TRAVERSIBLE_LO, [MOUNTAIN, MASSIF]), is_geographical_object)
    is_mountain_only = logical_negate(
        is_mountain_massif,
        [
            is_volcano
        ]
    )
    is_physical_object_only = logical_negate(
        is_physical_object,
        [
            is_audio_visual_work,
            is_art_work,
            is_musical_work,
            is_geographical_object,
            is_currency,
            is_gas,
            is_clothing,
            is_chemical_compound,
            is_electromagnetic_wave,
            is_song,
            is_food,
            is_character,
            is_law,
            is_software,
            is_website,
            is_vehicle,
            is_lake,
            is_landform,
            is_railroad,
            is_airport,
            is_aircraft,
            is_watercraft,
            is_sex_toy,
            is_data_format,
            is_date,
            is_research_method,
            is_sport,
            is_watercraft,
            is_aircraft,
            is_brand,
            is_vehicle_brand,
            is_road_vehicle,
            is_railroad,
            is_radio_program,
            is_weapon,
            is_book_magazine_article_proverb,
            is_brand,
            is_organization,
            is_facility,
            is_anatomical_structure,
            is_gene,
            is_monument
        ]
    )
    is_musical_work_only = logical_negate(
        is_musical_work,
        [
            is_song
        ]
    )
    is_geographical_object_only = logical_negate(
        is_geographical_object,
        [
            is_river,
            is_lake,
            is_sea,
            is_volcano,
            is_mountain_only,
            is_region,
            is_monument,
            is_country,
            is_facility,
            is_food,
            is_airport,
            is_bridge,
            is_train_station
        ]
    )

    is_event_election_only = logical_negate(
        logical_ors([is_event, is_election, is_accident]),
        [
            is_award_ceremony,
            is_war,
            is_natural_phenomenon
        ]
    )
    is_region_only = logical_negate(
        is_region,
        [
            is_populated_place,
            is_country,
            is_lake,
            is_river,
            is_sea,
            is_volcano,
            is_mountain_only
        ]
    )
    is_astronomical_object_only = logical_negate(
        is_astronomical_object,
        [
            is_geographical_object
        ]
    )

    is_date_only = logical_negate(
        is_date,
        [
            is_strategy,
            is_development_biology
        ]
    )
    is_development_biology_date = logical_and(is_development_biology, is_date)
    is_value_only = logical_negate(
        is_value,
        [
            is_unit_of_mass,
            is_event,
            is_election,
            is_currency,
            is_number,
            is_physical_quantity,
            is_award,
            is_date,
            is_postal_code
        ]
    )
    is_activity_subclass_only = logical_negate(
        logical_or(is_activity_subclass, is_activity),
        [
            is_crime,
            is_war,
            is_chemical_compound,
            is_gene,
            is_molecule,
            is_mathematical_object,
            is_sport,
            is_sport_event,
            is_event,
            is_paradigm,
            is_position,
            is_title,
            is_algorithm,
            is_order,
            is_organization,
            is_research_method,
            is_proposition,
            is_taxonomic_rank,
            is_algorithm,
            is_event,
            is_election,
            is_genre,
            is_concept
        ]
    )
    is_crime_only = logical_negate(
        is_crime,
        [
            is_war
        ]
    )
    is_number_only = logical_negate(
        is_number,
        [
            is_physical_quantity
        ]
    )
    is_molecule_only = logical_negate(
        is_molecule,
        [
            is_gene,
            is_chemical_compound
        ]
    )
    # VEHICLES:
    is_vehicle_only = logical_negate(
        is_vehicle,
        [
            is_watercraft,
            is_aircraft,
            is_road_vehicle
        ]
    )
    is_watercraft_only = logical_negate(
        is_watercraft,
        [
            is_aircraft
        ]
    )
    is_road_vehicle_only = logical_negate(
        is_road_vehicle,
        [
            is_aircraft,
            is_watercraft,
        ]
    )
    # remove groups that have occupations from mathematical objects:
    is_object_with_occupation = c.satisfy([wprop.INSTANCE_OF, wprop.OCCUPATION], [OCCUPATION, PROFESSION, POSITION])
    is_mathematical_object_only = logical_negate(
        is_mathematical_object,
        [
            is_geometric_shape,
            is_physical_quantity,
            is_number,
            is_object_with_occupation,
            is_landform
        ]
    )
    is_organization_only = logical_negate(
        is_organization,
        [
            is_country,
            is_geographical_object,
            is_family,
            is_people
        ]
    )
    is_art_work_only = logical_negate(
        is_art_work,
        [
            is_musical_work,
            is_audio_visual_work,
            is_sex_toy,
            is_monument
        ]
    )
    is_software_only = logical_negate(
        is_software,
        [
            is_language,
            is_organization,
            is_website
        ]
    )
    is_website_only = logical_negate(
        is_website,
        [
            is_organization,
            is_language
        ]
    )
    is_taxon_or_breed_only = logical_negate(
        is_taxon_or_breed,
        [
            is_human,
            is_plant
        ]
    )
    is_human_only = logical_negate(
        is_human,
        [
            is_male,
            is_female,
            is_kin,
            is_kinship,
            is_title
        ]
    )
    is_weapon_only = logical_negate(
        is_weapon,
        [
            is_software,
            is_website,
            is_vehicle
        ]
    )
    is_book_magazine_article_proverb_only = logical_negate(
        is_book_magazine_article_proverb,
        [
            is_software,
            is_website,
            is_musical_work,
            is_song,
            is_law,
            is_legal_action
        ]
    )
    is_fictional_character_only = logical_negate(
        is_fictional_character,
        [
            is_human,
            is_stock_character
        ]
    )
    is_battle_only = logical_negate(
        is_battle,
        [
            is_war,
            is_crime
        ]
    )
    is_brand_only = logical_negate(
        is_brand,
        [
            is_vehicle,
            is_aircraft,
            is_watercraft,
            is_website,
            is_software,
            is_vehicle_brand
        ]
    )
    is_vehicle_brand_only = logical_negate(
        is_vehicle_brand,
        [
            is_vehicle,
            is_aircraft,
            is_watercraft,
            is_website,
            is_software
        ]
    )
    is_concept_paradigm_proposition_only = logical_negate(
        logical_ors([is_concept, is_paradigm, is_proposition]),
        [
            is_physical_object,
            is_physical_quantity,
            is_software,
            is_website,
            is_color,
            is_vehicle,
            is_electromagnetic_wave,
            is_brand,
            is_vehicle_brand,
            is_currency,
            is_fictional_character,
            is_human,
            is_aircraft,
            is_geographical_object,
            is_geometric_shape,
            is_mathematical_object,
            is_musical_work,
            is_mountain_massif,
            is_lake,
            is_landform,
            is_language,
            is_anatomical_structure,
            is_book_magazine_article_proverb,
            is_development_biology,
            is_plant,
            is_sexual_orientation,
            is_genre,
            is_legislative_term
        ]
    )
    is_anatomical_structure_only = logical_negate(
        is_anatomical_structure,
        [
            is_plant
        ]
    )
    is_facility_only = logical_negate(
        is_facility,
        [
            is_train_station,
            is_aircraft,
            is_airport,
            is_bridge,
            is_vehicle,
            is_astronomical_object,
            is_railroad,
            is_monument
        ]
    )
    is_wikipedia_list_only = logical_negate(
        is_wikipedia_list,
        [
            is_activity_subclass,
            is_alphabet,
            is_art_work,
            is_astronomical_object,
            is_audio_visual_work,
            is_award,
            is_character,
            is_character,
            is_chemical_compound,
            is_color,
            is_currency,
            is_disease,
            is_election,
            is_electromagnetic_wave,
            is_facility,
            is_fictional_character,
            is_gene,
            is_genre,
            is_geographical_object,
            is_human,
            is_language,
            is_law,
            is_law,
            is_legal_action,
            is_legal_case,
            is_legislative_term,
            is_mathematical_object,
            is_mind,
            is_people,
            is_person,
            is_person,
            is_physical_object,
            is_populated_place,
            is_position,
            is_region,
            is_religion,
            is_research_method,
            is_sexual_orientation,
            is_software,
            is_speech,
            is_sport,
            is_sport_event,
            is_stock_character,
            is_strategy,
            is_taxon_or_breed,
            is_value,
            is_vehicle,
            is_wikidata_prop,
            is_weapon
        ]
    )
    is_sport_only = logical_negate(
        is_sport,
        [
            is_sport_event
        ]
    )
    is_legal_action_only = logical_negate(
        is_legal_action,
        [
            is_law,
            is_election
        ]
    )
    is_genre_only = logical_negate(
        is_genre,
        [
            is_physical_object,
            is_audio_visual_work,
            is_art_work,
            is_book_magazine_article_proverb,
            is_concept
        ]
    )
    is_plant_only = logical_negate(
        is_plant,
        [
            is_food,
            is_human,
            is_organization
        ]
    )
    is_kinship_kin_only = logical_negate(
        logical_or(is_kinship, is_kin),
        [
            is_family
        ]
    )
    is_position_only = logical_negate(
        is_position,
        [
            is_organization,
            is_human
        ]
    )
    is_radio_program_only = logical_negate(
        is_radio_program,
        [
            is_audio_visual_work,
        ]
    )
    is_taxonomic_rank_only = logical_negate(
        is_taxonomic_rank,
        [
            is_order
        ]
    )
    is_research_method_only = logical_negate(
        is_research_method,
        [
            is_audio_visual_work,
            is_book_magazine_article_proverb,
            is_art_work,
            is_concept,
            is_crime,
            is_war,
            is_algorithm,
            is_law,
            is_legal_action,
            is_legal_case
        ]
    )
    is_algorithm_only = logical_negate(
        is_algorithm,
        [
            is_concept,
            is_paradigm
        ]
    )

    is_union_only = logical_negate(
        is_union,
        [
            is_kinship,
            is_human,
            is_person
        ]
    )
    # get all the wikidata items that are disconnected:
    no_instance_subclass_or_cat_link = logical_ands(
        [
            c.relation(relation_name).edges() == 0
            for relation_name in [wprop.PART_OF, wprop.INSTANCE_OF, wprop.SUBCLASS_OF, wprop.CATEGORY_LINK]
        ]
    )
    is_sports_terminology_only = logical_negate(
        is_sports_terminology,
        [
            is_organization,
            is_human,
            is_person,
            is_activity,
            is_title,
            is_physical_quantity
        ]
    )

    out = {
        "aaa_wikidata_prop": is_wikidata_prop,
        "aaa_wikipedia_disambiguation": is_wikipedia_disambiguation,
        "aaa_wikipedia_template_namespace": is_wikipedia_template_namespace,
        "aaa_wikipedia_user_language_template": is_wikipedia_user_language_template,
        "aaa_wikipedia_list": is_wikipedia_list_only,
        "aaa_wikipedia_project_page": is_wikipedia_project_page,
        "aaa_wikimedia_category_page": is_wikimedia_category_page,
        "aaa_no_instance_subclass_or_link": no_instance_subclass_or_cat_link,
        "taxon": is_taxon_or_breed_only,
        "human_male": is_human_male,
        "human_female": is_human_female,
        "human": is_human_only,
        "fictional_character": is_fictional_character_only,
        "people": is_people,
        "language": is_language_only,
        "alphabet": is_alphabet_only,
        "speech": is_speech,
        "gas": is_gas_only,
        "gene": is_gene,
        "molecule": is_molecule_only,
        "astronomical_object": is_astronomical_object_only,
        "disease": is_disease,
        "mind": is_mind,
        "song": is_song,
        "radio_program": is_radio_program_only,
        "law": is_law,
        "legal_action": is_legal_action_only,
        "book_magazine_article": is_book_magazine_article_proverb_only,
        "chemical_compound": is_chemical_compound_only,
        "geometric_shape": is_geometric_shape,
        "mathematical_object": is_mathematical_object_only,
        "physical_quantity": is_physical_quantity,
        "number": is_number_only,
        "geographical_object": is_geographical_object_only,
        "train_station": is_train_station,
        "railroad": is_railroad,
        "concept": is_concept_paradigm_proposition_only,
        "genre": is_genre_only,
        "sexual_orientation": is_sexual_orientation,
        "bridge": is_bridge,
        "airport": is_airport,
        "river": is_river,
        "lake": is_lake,
        "sea": is_sea,
        "weapon": is_weapon_only,
        "region": is_region_only,
        "country": is_country,
        "software": is_software_only,
        "website": is_website_only,
        "volcano": is_volcano,
        "mountain": is_mountain_only,
        "religion": is_religion,
        "organization": is_organization_only,
        "musical_work": is_musical_work_only,
        "other_art_work": is_art_work_only,
        "audio_visual_work": is_audio_visual_work,
        "physical_object": is_physical_object_only,
        "record_chart": is_record_chart,
        "clothing": is_clothing,
        "plant": is_plant_only,
        "anatomical_structure": is_anatomical_structure_only,
        "facility": is_facility_only,
        "monument": is_monument,
        "vehicle": is_vehicle_only,
        "watercraft": is_watercraft_only,
        "road_vehicle": is_road_vehicle_only,
        "vehicle_brand": is_vehicle_brand_only,
        "brand": is_brand_only,
        "aircraft": is_aircraft,
        "legal_case": is_legal_case,
        "position": is_position_only,
        "person_role": is_person_only,
        "populated_place": is_populated_place,
        "value": is_value_only,
        "unit_of_mass": is_unit_of_mass,
        "currency": is_currency,
        "postal_code": is_postal_code,
        "name": is_name,
        "data_format": is_data_format,
        "character": is_character,
        "family": is_family,
        "sport": is_sport_only,
        "taxonomic_rank": is_taxonomic_rank,
        "sex_toy": is_sex_toy,
        "legislative_term": is_legislative_term,
        "sport_event": is_sport_event,
        "date": is_date_only,
        "kinship": is_kinship_kin_only,
        "union": is_union_only,
        "research": is_research_method_only,
        "title": is_title,
        "hazard": is_hazard,
        "color": is_color,
        "sports_terminology": is_sports_terminology_only,
        "developmental_biology_period": is_development_biology_date,
        "strategy": is_strategy,
        "event": is_event_election_only,
        "natural_phenomenon": is_natural_phenomenon,
        "electromagnetic_wave": is_electromagnetic_wave,
        "war": is_war,
        "award": is_award,
        "crime": is_crime_only,
        "battle": is_battle_only,
        "international_relations": is_international_relations,
        "food": is_food,
        "algorithm": is_algorithm,
        "activity": is_activity_subclass_only,
        "award_ceremony": is_award_ceremony
    }
    # is_other = logical_not(logical_ors([val for key, val in out.items() if key != "aaa_wikipedia_list"]))
    # c.class_report([wprop.IS_A_LIST_OF, wprop.CATEGORY_LINK], logical_and(
    #     is_other,
    #     is_wikipedia_list_only
    # ), name="remaining lists")
    return out
