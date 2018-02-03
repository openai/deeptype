"""
Obtain a finer-grained classification of places and entities according to their associated
country/region.
"""
from numpy import (
    logical_and, logical_or, logical_not, logical_xor, where
)
from wikidata_linker_utils.logic import logical_negate, logical_ors
import wikidata_linker_utils.wikidata_properties as wprop


def wkp(c, name):
    """Convert a string wikipedia article name to its Wikidata index."""
    return c.article2id["enwiki/" + name][0][0]

def wkd(c, name):
    """Convert a wikidata QID to its wikidata index."""
    return c.name2index[name]


def classify(c):
    TRAVERSIBLE_BASIC =   [wprop.INSTANCE_OF, wprop.SUBCLASS_OF]
    TRAVERSIBLE_COUNTRY = [
        wprop.INSTANCE_OF,
        wprop.SUBCLASS_OF,
        wprop.COUNTRY_OF_CITIZENSHIP,
        wprop.COUNTRY,
        wprop.LOCATION,
        wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY
    ]
    TRAVERSIBLE_PART_OF = [
        wprop.INSTANCE_OF,
        wprop.SUBCLASS_OF,
        wprop.CONTINENT,
        wprop.PART_OF,
        wprop.COUNTRY_OF_CITIZENSHIP,
        wprop.COUNTRY,
        wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY
    ]
    TRAVERSIBLE_TOPIC =  [
        wprop.INSTANCE_OF, wprop.SUBCLASS_OF,
        wprop.STUDIES, wprop.FIELD_OF_THIS_OCCUPATION, wprop.OCCUPATION,
        wprop.FIELD_OF_WORK, wprop.INDUSTRY]

    ASSOCIATION_FOOTBALL_PLAYER = wkd(c,"Q937857")
    PAINTER = wkd(c,"Q1028181")
    POLITICIAN = wkd(c,"Q82955")
    ARTICLE = wkd(c,"Q191067")
    VIDEO_GAME = wkd(c,"Q7889")
    FILM = wkd(c,"Q11424")
    FICTIONAL_CHARACTER = wkd(c,"Q95074")
    POEM = wkd(c,"Q482")
    BOOK = wkd(c,"Q571")
    DISEASE = wkd(c,"Q12136")
    PAINTING = wkd(c,"Q3305213")
    VISUAL_ART_WORK = wkd(c,"Q4502142")
    MUSIC_WORK = wkd(c,"Q2188189")
    SCIENTIFIC_ARTICLE = wkd(c,"Q13442814")
    PROTEIN_FAMILY = wkd(c,"Q417841")
    PROTEIN_COMPLEX = wkd(c,"Q420927")
    GENE = wkd(c,"Q7187")
    CHEMICAL_SUBSTANCE = wkd(c,"Q79529")
    PROTEIN = wkd(c,"Q8054")
    TAXON = wkd(c,"Q16521")
    PHYSICAL_OBJECT = wkd(c,"Q223557")
    OUTERSPACE = wkp(c, 'Astronomical object')
    #INTERNATIONAL_ORGANISATION = wkd(c,"")
    HUMAN = wkp(c,"Human")
    HUMAN_SETTLMENT = wkd(c,"Q486972")
    DICTIONARY = wkd(c,"Q23622")
    ABRREVIATION =  wkd(c,"Q102786")
    POPULATED_PLACE = wkd(c,"Q486972")
    TERRITORIAL_ENTITY = wkd(c, "Q1496967")
    DESA = wkd(c,"Q26211545")
    TOWN_IN_CHINA = wkd(c,"Q735428")
    ADMIN_DIVISION_CHINA = wkd(c,"Q50231")
    COUNTRY = wkd(c,"Q6256")
    MOUNTAIN_RANGE = wkd(c,"Q46831")
    EARTH = wkp(c, "Earth")
    GEOGRAPHIC_LOCATION = wkd(c, "Q2221906")

    is_politician = c.satisfy([wprop.OCCUPATION], [POLITICIAN])
    is_painter = c.satisfy([wprop.OCCUPATION], [PAINTER])
    is_association_football_player = c.satisfy([wprop.OCCUPATION],[ASSOCIATION_FOOTBALL_PLAYER])

    is_populated_place  = c.satisfy(
        [wprop.INSTANCE_OF, wprop.PART_OF, wprop.CONTINENT, wprop.COUNTRY_OF_CITIZENSHIP,
        wprop.COUNTRY, wprop.SUBCLASS_OF],
        [GEOGRAPHIC_LOCATION, EARTH, HUMAN_SETTLMENT])

    is_taxon = c.satisfy(
        [wprop.INSTANCE_OF, wprop.PART_OF,  wprop.SUBCLASS_OF],
        [TAXON])
    is_other_wkd= c.satisfy(
        [wprop.INSTANCE_OF, wprop.PART_OF, wprop.SUBCLASS_OF],
        [GENE, CHEMICAL_SUBSTANCE, SCIENTIFIC_ARTICLE,
         PROTEIN,   DISEASE, PROTEIN_FAMILY,PROTEIN_COMPLEX,
         BOOK, MUSIC_WORK, PAINTING, VISUAL_ART_WORK, POEM, FILM,
         FICTIONAL_CHARACTER,VIDEO_GAME,SCIENTIFIC_ARTICLE,ARTICLE])
    is_gene_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Genes")], max_steps=5)
    is_chromosome_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Chromosomes")], max_steps=5)
    is_protein_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Proteins")], max_steps=5)
    is_other= logical_ors([is_other_wkd, is_gene_wkp, is_chromosome_wkp,
               is_protein_wkp ])




    print("WIKI Links")
    WIKIPEDIA_DISAMBIGUATION_PAGE = wkd(c,"Q4167410")
    SCIENTIFIC_JOURNAL = wkd(c,"Q5633421")
    SURNAME = wkd(c,"Q101352")
    WIKI_NEWS_ARTICLE = wkd(c,"Q17633526")
    WIKIMEDIA_CATEGORY = wkd(c,"Q4167836")
    WIKIPEDIA_TEMPLATE_NAMESPACE = wkd(c,"Q11266439")
    WIKIPEDIA_LIST = wkd(c,"Q13406463")
    ENCYCLOPEDIA_ARTICLE = wkd(c,"Q17329259")
    WIKIMEDIA_PROJECT_PAGE = wkd(c,"Q14204246")
    RURAL_COMUNE_VIETNAM = wkd(c,"Q2389082")
    TERRITORIAL_ENTITY = wkd(c,"Q1496967")
    is_Wiki_Links = c.satisfy(TRAVERSIBLE_TOPIC,
        [WIKIPEDIA_DISAMBIGUATION_PAGE,
         SURNAME,
         WIKIMEDIA_CATEGORY,
         WIKIPEDIA_TEMPLATE_NAMESPACE,
         WIKIPEDIA_LIST,
         ENCYCLOPEDIA_ARTICLE,
         WIKIMEDIA_PROJECT_PAGE,
         WIKI_NEWS_ARTICLE
          ])


    print("is_in_outer_space")
    is_in_outer_space = c.satisfy(TRAVERSIBLE_PART_OF, [OUTERSPACE])
    print("part_of_earth")
    part_of_earth = c.satisfy(
        [wprop.INSTANCE_OF, wprop.PART_OF, wprop.CONTINENT, wprop.COUNTRY_OF_CITIZENSHIP, wprop.COUNTRY, wprop.SUBCLASS_OF, wprop.LOCATION],
        [GEOGRAPHIC_LOCATION, EARTH])
    print("is_in_outer_space_not_earth")
    is_in_outer_space_not_earth = logical_negate(
        is_in_outer_space, [part_of_earth])


    print("African countries")
    ALGERIA = wkp(c,"Algeria")
    ANGOLA = wkp(c,"Angola")
    BENIN = wkp(c,"Benin")
    BOTSWANA = wkd(c,"Q963")
    BURKINA_FASO = wkd(c,"Q965")
    BURUNDI = wkd(c,"Q967")
    CAMEROON = wkd(c,"Q1009")
    CAPE_VERDE = wkd(c,"Q1011")
    CHAD = wkd(c,"Q657")
    CENTRAL_AFRICAN_REPUBLIC = wkd(c,"Q929")
    COMOROS = wkd(c,"Q970")
    DEMOCRATIC_REPUBLIC_OF_CONGO = wkd(c,"Q974")
    REPUBLIC_OF_CONGO = wkd(c,"Q971")
    DJIBOUTI = wkd(c,"Q977")
    EGYPT = wkd(c,"Q79")
    RASHIDUN_CALIPHATE = wkd(c,"Q12490507")
    EQUATORIAL_GUINEA = wkd(c,"Q983")
    ERITREA = wkd(c,"Q986")
    ETHIOPIA = wkd(c,"Q115")
    GABON = wkd(c,"Q1000")
    THE_GAMBIA = wkd(c,"Q1005")
    GHANA = wkd(c,"Q117")
    GUINEA = wkd(c,"Q1006")
    GUINEA_BISSAU = wkd(c,"Q1007")
    IVORY_COAST = wkd(c,"Q1008")
    KENYA = wkd(c,"Q114")
    LESOTHO = wkd(c,"Q1013")
    LIBERIA = wkd(c,"Q1014")
    LIBYA = wkd(c,"Q1016")
    MADAGASCAR = wkd(c,"Q1019")
    MALAWI = wkd(c,"Q1020")
    MALI = wkd(c,"Q912")
    MAURITANIA = wkd(c,"Q1025")
    MAURITIUS = wkd(c,"Q1027")
    MOROCCO = wkd(c,"Q1028")
    MOZAMBIQUE = wkd(c,"Q1029")
    NAMIBIA = wkd(c,"Q1030")
    NIGER = wkd(c,"Q1032")
    NIGERIA = wkd(c,"Q1033")
    RWANDA = wkd(c,"Q1037")
    SAHARI_ARAB_DEOMOCRATIC_REPUBLIC = wkd(c,"Q40362")
    SAO_TOME_AND_PRINCIPE= wkd(c,"Q1039")
    SENEGAL = wkd(c,"Q1041")
    SEYCHELLES = wkd(c,"Q1042")
    SIERRA_LEONE = wkd(c,"Q1044")
    SOMALIA = wkd(c,"Q1045")
    SOUTH_AFRICA = wkd(c,"Q258")
    SOUTHSUDAN = wkd(c,"Q958")
    SUDAN = wkd(c,"Q1049")
    SWAZILAND= wkd(c,"Q1050")
    TANZANIA = wkd(c,"Q924")
    TOGO = wkd(c,"Q945")
    TUNISIA= wkd(c,"Q948")
    UGANDA = wkd(c,"Q1036")
    WESTERN_SAHARA = wkd(c,"Q6250")
    ZAMBIA = wkd(c,"Q953")
    ZIMBABWE = wkd(c,"Q954")
    SOMALI_LAND = wkd(c,"Q34754")


    in_algeria_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [ALGERIA])
    in_algeria_stubs = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Algeria stubs")], max_steps=4)
    in_algeria_politics = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Politics of Algeria")], max_steps=3)
    in_algeria_roads = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Roads in Algeria")], max_steps=3)
    in_algeria = logical_ors([in_algeria_wkd, in_algeria_stubs, in_algeria_politics, in_algeria_roads])
    in_angola_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [ANGOLA])
    in_angola_stubs = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Angola stubs")], max_steps=4)
    in_angola_politics = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Politics of Angola")], max_steps=3)
    in_angola_roads = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Roads in Angola")], max_steps=3)
    in_angola = logical_ors([in_angola_wkd , in_angola_stubs, in_angola_politics, in_angola_roads])
    in_benin_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [BENIN])
    in_benin_stubs = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Benin stubs")], max_steps=4)
    in_benin_politics = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Politics of Benin")], max_steps=3)
    in_benin_roads = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Roads in Benin")], max_steps=3)
    in_benin = logical_ors([in_benin_wkd, in_benin_stubs, in_benin_politics, in_benin_roads])
    in_botswana_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [BOTSWANA])
    in_botswana_stubs = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Botswana stubs")], max_steps=4)
    in_botswana_politics = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Politics of Botswana")], max_steps=3)
    in_botswana_roads = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Roads in Botswana")], max_steps=3)
    in_botswana = logical_ors([in_botswana_wkd, in_botswana_stubs, in_botswana_politics,in_botswana_roads])
    in_burkina_faso_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [BURKINA_FASO])
    in_bburkina_faso_stubs = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Burkina Faso stubs")], max_steps=4)
    in_bburkina_faso_politics = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Politics of Botswana")], max_steps=3)
    in_burkina_faso = logical_ors([in_burkina_faso_wkd , in_botswana_stubs, in_botswana_politics])
    in_burundi_politics_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Politics of Burkina Faso")], max_steps=4)
    in_burundi_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [BURUNDI])
    in_burundi = logical_ors([in_burundi_wkd,in_burundi_politics_wkp])
    in_cameroon = c.satisfy(TRAVERSIBLE_COUNTRY, [CAMEROON])
    in_cape_verde= c.satisfy(TRAVERSIBLE_COUNTRY, [CAPE_VERDE])
    in_chad = c.satisfy(TRAVERSIBLE_COUNTRY, [CHAD])
    in_central_african_republic = c.satisfy(TRAVERSIBLE_COUNTRY, [CENTRAL_AFRICAN_REPUBLIC])
    in_comoros = c.satisfy(TRAVERSIBLE_COUNTRY, [COMOROS])
    in_democratic_republic_congo = c.satisfy(TRAVERSIBLE_COUNTRY, [DEMOCRATIC_REPUBLIC_OF_CONGO])
    in_republic_of_congo = c.satisfy(TRAVERSIBLE_COUNTRY, [REPUBLIC_OF_CONGO])
    in_djibouti = c.satisfy(TRAVERSIBLE_COUNTRY, [DJIBOUTI])
    in_egypt_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [EGYPT])
    in_ancient_egypt =  c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Ancient Egypt")], max_steps=6)
    in_Rashidun_Caliphate = c.satisfy(TRAVERSIBLE_COUNTRY, [RASHIDUN_CALIPHATE])
    egyptian_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Egyptian people")], max_steps=6)
    in_egypt = logical_ors([in_egypt_wkd, in_egypt_wkd,in_Rashidun_Caliphate, egyptian_people])
    in_equatorial_guinea = c.satisfy(TRAVERSIBLE_COUNTRY, [EQUATORIAL_GUINEA])
    in_eritrea = c.satisfy(TRAVERSIBLE_COUNTRY, [ERITREA])
    in_ethiopia = c.satisfy(TRAVERSIBLE_COUNTRY, [ETHIOPIA])
    in_gabon = c.satisfy(TRAVERSIBLE_COUNTRY, [GABON])
    in_the_gambia = c.satisfy(TRAVERSIBLE_COUNTRY, [THE_GAMBIA])
    in_ghana = c.satisfy(TRAVERSIBLE_COUNTRY, [GHANA])
    in_guinea = c.satisfy(TRAVERSIBLE_COUNTRY, [GUINEA])
    in_guinea_bissau = c.satisfy(TRAVERSIBLE_COUNTRY, [GUINEA_BISSAU])
    in_ivory_coast = c.satisfy(TRAVERSIBLE_COUNTRY, [IVORY_COAST])
    in_lesotho = c.satisfy(TRAVERSIBLE_COUNTRY, [LESOTHO])
    in_kenya = c.satisfy(TRAVERSIBLE_COUNTRY, [KENYA])
    in_liberia = c.satisfy(TRAVERSIBLE_COUNTRY, [LIBERIA])
    in_libya = c.satisfy(TRAVERSIBLE_COUNTRY, [LIBYA])
    in_madagascar = c.satisfy(TRAVERSIBLE_COUNTRY, [MADAGASCAR])
    in_malawi = c.satisfy(TRAVERSIBLE_COUNTRY, [MALAWI])
    in_mali = c.satisfy(TRAVERSIBLE_COUNTRY, [MALI])
    in_mauritania = c.satisfy(TRAVERSIBLE_COUNTRY, [MAURITANIA])
    in_mauritius = c.satisfy(TRAVERSIBLE_COUNTRY, [MAURITIUS])
    in_morrocco = c.satisfy(TRAVERSIBLE_COUNTRY, [MOROCCO])
    in_mozambique = c.satisfy(TRAVERSIBLE_COUNTRY, [MOZAMBIQUE])
    in_namibia = c.satisfy(TRAVERSIBLE_COUNTRY, [NAMIBIA])
    in_niger = c.satisfy(TRAVERSIBLE_COUNTRY, [NIGER])
    in_nigeria = c.satisfy(TRAVERSIBLE_COUNTRY, [NIGERIA])
    in_rwanda = c.satisfy(TRAVERSIBLE_COUNTRY, [RWANDA])
    in_sadr = c.satisfy(TRAVERSIBLE_COUNTRY, [SAHARI_ARAB_DEOMOCRATIC_REPUBLIC])
    in_stap = c.satisfy(TRAVERSIBLE_COUNTRY, [SAO_TOME_AND_PRINCIPE])
    in_senegal = c.satisfy(TRAVERSIBLE_COUNTRY, [SENEGAL])
    in_seychelles = c.satisfy(TRAVERSIBLE_COUNTRY, [SEYCHELLES])
    in_sierra_leone = c.satisfy(TRAVERSIBLE_COUNTRY, [SIERRA_LEONE])
    in_somalia = c.satisfy(TRAVERSIBLE_COUNTRY, [SOMALIA])
    in_somali_land = c.satisfy(TRAVERSIBLE_COUNTRY, [SOMALI_LAND])
    in_south_africa = c.satisfy(TRAVERSIBLE_COUNTRY, [SOUTH_AFRICA])
    in_ssudan= c.satisfy(TRAVERSIBLE_COUNTRY, [SOUTHSUDAN])
    in_sudan= c.satisfy(TRAVERSIBLE_COUNTRY, [SUDAN])
    in_swaziland= c.satisfy(TRAVERSIBLE_COUNTRY, [SWAZILAND])
    in_tanzania_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Sports competitions in Tanzania")], max_steps=4)
    in_tanzania_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [TANZANIA])
    in_tanzania = logical_ors([in_tanzania_wkp,in_tanzania_wkd])
    in_togo = c.satisfy(TRAVERSIBLE_COUNTRY, [TOGO])
    in_tunisia = c.satisfy(TRAVERSIBLE_COUNTRY, [TUNISIA])
    in_uganda = c.satisfy(TRAVERSIBLE_COUNTRY, [UGANDA])
    in_western_sahara = c.satisfy(TRAVERSIBLE_COUNTRY, [WESTERN_SAHARA])
    in_zambia_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [ZAMBIA])
    zambian_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Zambian people")], max_steps=4)
    in_zambia = logical_ors([in_zambia_wkd, zambian_people])
    in_zimbabwe = c.satisfy(TRAVERSIBLE_COUNTRY, [ZIMBABWE])
    in_africa = logical_ors([
        in_botswana,
        in_burkina_faso,
        in_burundi,
        in_cameroon,
        in_cape_verde,
        in_chad,
        in_central_african_republic,
        in_comoros,
        in_democratic_republic_congo,
        in_republic_of_congo,
        in_djibouti,
        in_egypt,
        in_equatorial_guinea,
        in_eritrea,
        in_ethiopia,
        in_gabon,
        in_the_gambia,
        in_ghana,
        in_guinea,
        in_guinea_bissau,
        in_ivory_coast,
        in_lesotho,
        in_kenya,
        in_liberia,
        in_libya,
        in_madagascar,
        in_malawi
        ])

    print("Oceanian countries")
    AUSTRALIA = wkd(c,"Q408")
    FIJI = wkd(c,"Q712")
    INDONESIA = wkd(c,"Q252")
    KIRIBATI= wkd(c,"Q710")
    MARSHALL_ISLANDS= wkd(c,"Q709")
    FEDERATED_STATES_OF_MICRONESIA= wkd(c,"Q702")
    NAURU= wkd(c,"Q697")
    PALAU= wkd(c,"Q695")
    PAPUA_NEW_GUINEA= wkd(c,"Q691")
    SAMOA = wkd(c,"Q683")
    SOLOMON_ISLANDS= wkd(c,"Q685")
    VANUATU = wkd(c,"Q686")
    NEW_ZEALAND = wkd(c,"Q664")

    in_australia_athletes = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Australian sportspeople")], max_steps=5)
    in_australia_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [AUSTRALIA])
    in_australia = logical_ors([in_australia_wkd, in_australia_athletes])
    in_fiji = c.satisfy(TRAVERSIBLE_COUNTRY, [FIJI])
    in_indonesia = c.satisfy(TRAVERSIBLE_COUNTRY, [INDONESIA])
    in_kiribati = c.satisfy(TRAVERSIBLE_COUNTRY, [KIRIBATI])
    in_marshall_islands = c.satisfy(TRAVERSIBLE_COUNTRY, [MARSHALL_ISLANDS])
    in_federates_states_of_micronesia = c.satisfy(TRAVERSIBLE_COUNTRY, [FEDERATED_STATES_OF_MICRONESIA])
    in_nauru = c.satisfy(TRAVERSIBLE_COUNTRY, [NAURU])
    in_palau = c.satisfy(TRAVERSIBLE_COUNTRY, [PALAU])
    in_papua_new_guinea =  c.satisfy(TRAVERSIBLE_COUNTRY, [PAPUA_NEW_GUINEA])
    in_samoa_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Samoa")], max_steps=5)
    in_samoa_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [SAMOA])
    in_samoa = logical_ors([in_samoa_wkd, in_samoa_wkp])
    in_solomon_islands = c.satisfy(TRAVERSIBLE_COUNTRY, [SOLOMON_ISLANDS])
    in_vanuatu = c.satisfy(TRAVERSIBLE_COUNTRY, [VANUATU])
    in_new_zealand = c.satisfy(TRAVERSIBLE_COUNTRY, [NEW_ZEALAND])

    print("South American countries")
    ARGENTINA = wkd(c,"Q414")
    BOLIVIA = wkd(c,"Q750")
    BRAZIL = wkd(c,"Q155")
    CHILE = wkd(c,"Q298")
    COLOMBIA = wkd(c,"Q739")
    ECUADOR = wkd(c,"Q736")
    GUYANA = wkd(c,"Q734")
    PARAGUAY = wkd(c,"Q733")
    PERU = wkd(c,"Q419")
    SURINAME = wkd(c,"Q730")
    TRINIDAD_AND_TOBAGO = wkd(c,"Q754")
    URUGUAY = wkd(c,"Q77")
    VENEZUELA = wkd(c,"Q717")

    in_argentina = c.satisfy(TRAVERSIBLE_COUNTRY, [ARGENTINA])
    in_bolivia = c.satisfy(TRAVERSIBLE_COUNTRY, [BOLIVIA])
    in_brazil = c.satisfy(TRAVERSIBLE_COUNTRY, [BRAZIL])
    in_chile = c.satisfy(TRAVERSIBLE_COUNTRY, [CHILE])
    in_colombia = c.satisfy(TRAVERSIBLE_COUNTRY, [COLOMBIA])
    in_ecuador = c.satisfy(TRAVERSIBLE_COUNTRY, [ECUADOR])
    in_guyana = c.satisfy(TRAVERSIBLE_COUNTRY, [GUYANA])
    in_paraguay = c.satisfy(TRAVERSIBLE_COUNTRY, [PARAGUAY])
    in_peru = c.satisfy(TRAVERSIBLE_COUNTRY, [PERU])
    in_suriname = c.satisfy(TRAVERSIBLE_COUNTRY, [SURINAME])
    in_trinidad_and_tobago = c.satisfy(TRAVERSIBLE_COUNTRY, [TRINIDAD_AND_TOBAGO])
    in_uruguay = c.satisfy(TRAVERSIBLE_COUNTRY, [URUGUAY])
    in_venezuela = c.satisfy(TRAVERSIBLE_COUNTRY, [VENEZUELA])

    print("Central American countries")
    BELIZE = wkd(c,"Q242")
    COSTA_RICA = wkd(c,"Q800")
    EL_SALVADOR = wkd(c,"Q792")
    GUATEMALA = wkd(c,"Q774")
    HONDURAS = wkd(c,"Q783")
    NICARAGUA = wkd(c,"Q811")
    PANAMA = wkd(c,"Q804")

    in_belize = c.satisfy(TRAVERSIBLE_COUNTRY, [BELIZE])
    in_costa_rica = c.satisfy(TRAVERSIBLE_COUNTRY, [COSTA_RICA])
    in_el_salvador = c.satisfy(TRAVERSIBLE_COUNTRY, [EL_SALVADOR])
    in_guatemala = c.satisfy(TRAVERSIBLE_COUNTRY, [GUATEMALA])
    in_honduras = c.satisfy(TRAVERSIBLE_COUNTRY, [HONDURAS])
    in_nicaragua = c.satisfy(TRAVERSIBLE_COUNTRY, [NICARAGUA])
    in_panama = c.satisfy(TRAVERSIBLE_COUNTRY, [PANAMA])

    print("North American countries")
    ANTIGUA_BARBUDA = wkd(c,"Q781")
    BAHAMAS = wkd(c,"Q778")
    BARBADOS = wkd(c,"Q244")
    BELIZE = wkd(c,"Q242")
    CANADA = wkd(c,"Q16")
    COSTA_RICA = wkd(c,"Q800")
    CUBA = wkd(c,"Q241")
    DOMINICAN_REPUBLIC = wkd(c,"Q786")
    EL_SALVADOR = wkd(c,"Q792")
    GRENADA = wkd(c,"Q769")
    GUATEMALA = wkd(c,"Q774")
    HAITI = wkd(c,"Q790")
    HONDURAS = wkd(c,"Q783")
    JAMAICA = wkd(c,"Q766")
    MEXICO = wkd(c,"Q96")
    NICARAGUA = wkd(c,"Q811")
    PANAMA = wkd(c,"Q804")
    SAINT_KITTS_AND_NEVIS = wkd(c,"Q763")
    SAINT_LUCIA = wkd(c,"Q760")
    SAINT_VINCENT_AND_GRENADINES = wkd(c,"Q757")
    UNITED_STATES = wkd(c,"Q30")

    in_antigua_barbuda = c.satisfy(TRAVERSIBLE_COUNTRY, [ANTIGUA_BARBUDA])
    in_bahamas = c.satisfy(TRAVERSIBLE_COUNTRY, [BAHAMAS])
    in_barbados = c.satisfy(TRAVERSIBLE_COUNTRY, [BARBADOS])
    in_belize = c.satisfy(TRAVERSIBLE_COUNTRY, [BELIZE])
    canadians = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Canadian people by occupation")], max_steps=5)
    in_canada_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [CANADA])
    in_canada = logical_ors([canadians, in_canada_wkd])

    in_costa_rica = c.satisfy(TRAVERSIBLE_COUNTRY, [COSTA_RICA])
    in_cuba = c.satisfy(TRAVERSIBLE_COUNTRY, [CUBA])
    in_dominican_republic = c.satisfy(TRAVERSIBLE_COUNTRY, [DOMINICAN_REPUBLIC])
    in_el_salvador = c.satisfy(TRAVERSIBLE_COUNTRY, [EL_SALVADOR])
    in_grenada = c.satisfy(TRAVERSIBLE_COUNTRY, [GRENADA])
    in_guatemala = c.satisfy(TRAVERSIBLE_COUNTRY, [GUATEMALA])
    in_haiti = c.satisfy(TRAVERSIBLE_COUNTRY, [HAITI])
    in_honduras = c.satisfy(TRAVERSIBLE_COUNTRY, [HONDURAS])
    in_jamaica = c.satisfy(TRAVERSIBLE_COUNTRY, [JAMAICA])
    in_mexico = c.satisfy(TRAVERSIBLE_COUNTRY, [MEXICO])
    in_nicaragua = c.satisfy(TRAVERSIBLE_COUNTRY, [NICARAGUA])
    in_panama = c.satisfy(TRAVERSIBLE_COUNTRY, [PANAMA])
    in_Saint_Kitts_and_Nevis = c.satisfy(TRAVERSIBLE_COUNTRY, [SAINT_KITTS_AND_NEVIS])
    in_saint_lucia = c.satisfy(TRAVERSIBLE_COUNTRY, [SAINT_LUCIA])
    in_saint_vincent_and_grenadines = c.satisfy(TRAVERSIBLE_COUNTRY, [SAINT_VINCENT_AND_GRENADINES])
    in_usa_sports = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:History of sports in the United States")], max_steps=7)
    years_in_usa = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in the United States")], max_steps=7)
    in_usa_roads = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Roads in the United States")], max_steps=7)
    in_united_states_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [UNITED_STATES])
    in_united_states = logical_ors([in_usa_sports,in_united_states_wkd, years_in_usa])

    print("Asian countries")
    FOURTH_ADMIN_DIVISION_INDONESIA = wkd(c,"Q2225692")
    RURAL_COMUNE_VIETNAM = wkd(c,"Q2389082")
    AFGHANISTAN = wkd(c,"Q889")
    KINGDOM_OF_AFGHANISTAN = wkd(c,"Q1138904")
    REPUBLIC_OF_AFGHANISTAN = wkd(c,"Q1415128")
    DEMOCRATIC_REPUBLIC_OF_AFGHANISTAN = wkd(c,"Q476757")
    BANGLADESH = wkd(c,"Q902")
    BHUTAN = wkd(c,"Q917")
    BRUNEI = wkd(c,"Q921")
    CAMBODIA = wkd(c,"Q424")
    CHINA = wkd(c,"Q148")
    EAST_TIMOR = wkd(c,"Q574")
    INDIA = wkd(c,"Q668")
    INDONESIA = wkd(c,"Q252")
    IRAN = wkd(c,"Q794")
    IRAQ = wkd(c,"Q796")
    KURDISTAN = wkd(c,"Q41470")
    ISRAEL = wkd(c,"Q801")
    JAPAN = wkd(c,"Q17")
    JORDAN = wkd(c,"Q810")
    KAZAKHSTAN = wkd(c,"Q232")
    KUWAIT = wkd(c,"Q817")
    KYRGYZSTAN = wkd(c,"Q813")
    LAOS = wkd(c,"Q819")
    LEBANON = wkd(c,"Q822")
    MALAYSIA = wkd(c,"Q833")
    MALDIVES = wkd(c,"Q826")
    MONGOLIA = wkd(c,"Q711")
    MYANMAR = wkd(c,"Q836")
    NEPAL = wkd(c,"Q837")
    NORTH_KOREA = wkd(c,"Q423")
    OMAN = wkd(c,"Q842")
    PALESTINE = wkd(c,"Q219060")
    PAKISTAN = wkd(c,"Q843")
    PHILIPPINES = wkd(c,"Q928")
    QATAR = wkd(c,"Q846")
    SAUDI_ARABIA = wkd(c,"Q851")
    SINGAPORE = wkd(c,"Q334")
    SOUTH_KOREA = wkd(c,"Q884")
    SRI_LANKA = wkd(c,"Q854")
    SYRIA = wkd(c,"Q858")
    TAIWAN = wkd(c,"Q865")
    TAJIKISTAN = wkd(c,"Q863")
    THAILAND = wkd(c,"Q869")
    TURKMENISTAN = wkd(c,"Q874")
    UNITED_ARAB_EMIRATES = wkd(c,"Q878")
    UZBEKISTAN = wkd(c,"Q265")
    VIETNAM = wkd(c,"Q881")
    YEMEN =  wkd(c,"Q805")


    in_afghanistan = c.satisfy(TRAVERSIBLE_COUNTRY, [AFGHANISTAN, REPUBLIC_OF_AFGHANISTAN, DEMOCRATIC_REPUBLIC_OF_AFGHANISTAN])
    in_bangladesh = c.satisfy(TRAVERSIBLE_COUNTRY, [BANGLADESH])
    in_bhutan = c.satisfy(TRAVERSIBLE_COUNTRY, [BHUTAN])
    in_brunei = c.satisfy(TRAVERSIBLE_COUNTRY, [BRUNEI])
    in_cambodia = c.satisfy(TRAVERSIBLE_COUNTRY, [CAMBODIA])

    years_in_china = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in China")], max_steps=6)
    chinese_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Chinese people by occupation")], max_steps=6)
    is_tibetan_politician = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Tibetan politicians")], max_steps=6)
    in_china_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [CHINA])
    in_china = logical_ors([in_china_wkd,years_in_china,is_tibetan_politician, chinese_people])


    in_east_timor = c.satisfy(TRAVERSIBLE_COUNTRY, [EAST_TIMOR])
    in_india = c.satisfy(TRAVERSIBLE_COUNTRY, [INDIA])
    in_indonesia = c.satisfy(TRAVERSIBLE_COUNTRY, [INDONESIA,FOURTH_ADMIN_DIVISION_INDONESIA])
    in_iran = c.satisfy(TRAVERSIBLE_COUNTRY, [IRAN])
    in_iraq = c.satisfy(TRAVERSIBLE_COUNTRY, [IRAQ, KURDISTAN])
    in_israel = c.satisfy(TRAVERSIBLE_COUNTRY, [ISRAEL])
    in_japan = c.satisfy(TRAVERSIBLE_COUNTRY, [JAPAN])
    in_jordan = c.satisfy(TRAVERSIBLE_COUNTRY, [JORDAN])
    in_kazakhstan  = c.satisfy(TRAVERSIBLE_COUNTRY, [KAZAKHSTAN])
    in_kuwait = c.satisfy(TRAVERSIBLE_COUNTRY, [KUWAIT])
    in_kyrgyzstan = c.satisfy(TRAVERSIBLE_COUNTRY, [KYRGYZSTAN])
    in_laos = c.satisfy(TRAVERSIBLE_COUNTRY, [LAOS])
    in_lebanon = c.satisfy(TRAVERSIBLE_COUNTRY, [LEBANON])
    in_malaysia = c.satisfy(TRAVERSIBLE_COUNTRY, [MALAYSIA])
    in_maldives = c.satisfy(TRAVERSIBLE_COUNTRY, [MALDIVES])
    in_mongolia = c.satisfy(TRAVERSIBLE_COUNTRY, [MONGOLIA])
    in_myanmar = c.satisfy(TRAVERSIBLE_COUNTRY, [MYANMAR])
    in_nepal = c.satisfy(TRAVERSIBLE_COUNTRY, [NEPAL])
    in_north_korea = c.satisfy(TRAVERSIBLE_COUNTRY, [NORTH_KOREA])
    in_oman = c.satisfy(TRAVERSIBLE_COUNTRY, [OMAN])
    in_palestine = c.satisfy(TRAVERSIBLE_COUNTRY, [PALESTINE])
    in_pakistan =  c.satisfy(TRAVERSIBLE_COUNTRY, [PAKISTAN])
    in_philippines =  c.satisfy(TRAVERSIBLE_COUNTRY, [PHILIPPINES])
    in_qatar =  c.satisfy(TRAVERSIBLE_COUNTRY, [QATAR])
    in_saudi_arabia = c.satisfy(TRAVERSIBLE_COUNTRY, [SAUDI_ARABIA])
    in_singapore = c.satisfy(TRAVERSIBLE_COUNTRY, [SINGAPORE])
    in_south_korea_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [SOUTH_KOREA])
    korean_rulers = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Korean rulers")], max_steps=6)
    south_korea_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:South Korea")], max_steps=6)
    south_korean_rulers = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Korean rulers")], max_steps=6)
    in_south_korea = logical_ors([in_south_korea_wkd, korean_rulers])
    in_sri_lanka = c.satisfy(TRAVERSIBLE_COUNTRY, [SRI_LANKA])
    in_syria_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [SYRIA])
    ancient_syria = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Ancient Syria")], max_steps=6)
    in_syria = logical_ors([in_syria_wkd,ancient_syria])
    in_taiwan =  c.satisfy(TRAVERSIBLE_COUNTRY, [TAIWAN])
    in_tajikistan = c.satisfy(TRAVERSIBLE_COUNTRY, [TAJIKISTAN])
    in_thailand = c.satisfy(TRAVERSIBLE_COUNTRY, [THAILAND])
    in_turkmenistan = c.satisfy(TRAVERSIBLE_COUNTRY, [TURKMENISTAN])
    in_united_arab_emirates = c.satisfy(TRAVERSIBLE_COUNTRY, [UNITED_ARAB_EMIRATES])
    in_uzbekistan = c.satisfy(TRAVERSIBLE_COUNTRY, [UZBEKISTAN])
    in_vietnam = c.satisfy(TRAVERSIBLE_COUNTRY, [VIETNAM, RURAL_COMUNE_VIETNAM])
    in_yemen = c.satisfy(TRAVERSIBLE_COUNTRY, [YEMEN])


    print("European countries")
    ALBANIA = wkd(c,"Q222")
    ANDORRA = wkd(c,"Q228")
    ARMENIA = wkd(c,"Q399")
    AUSTRIA = wkd(c,"Q40")
    AUSTRIA_HUNGARY = wkd(c,"Q28513")
    AZERBAIJAN = wkd(c,"Q227")
    BELARUS = wkd(c,"Q184")
    BELGIUM = wkd(c,"Q31")
    BOSNIA = wkd(c,"Q225")
    BULGARIA = wkd(c,"Q219")
    CROATIA = wkd(c,"Q224")
    CYPRUS = wkd(c,"Q229")
    CZECH_REPUBLIC = wkd(c,"Q213")
    CZECHOSLOVAKIA = wkd(c,"Q33946")
    DENMARK = wkd(c,"Q35")
    ESTONIA = wkd(c,"Q191")
    FINLAND = wkd(c,"Q33")
    FRANCE = wkd(c,"Q142")
    GEORGIA = wkd(c,"Q230")
    GERMANY = wkd(c,"Q183")
    GERMANY_NAZI = wkd(c,"Q7318")
    GERMAN_EMPIRE = wkd(c,"Q43287")
    GERMAN_CONFEDERATION = wkd(c,"Q151624")
    EAST_GERMANY = wkd(c,"Q16957")
    GREECE = wkd(c,"Q41")
    HUNGARY = wkd(c,"Q28")
    ICELAND = wkd(c,"Q189")
    IRELAND = wkd(c,"Q27")
    ITALY = wkd(c,"Q38")
    ROMAN_EMPIRE = wkd(c,"Q2277")
    ANCIENT_ROME = wkd(c,"Q1747689")
    KINGDOM_OF_ITALY = wkd(c,"Q172579")
    NATIONAL_FASCIST_PARTY = wkd(c,"Q139596")
    KAZAKHSTAN = wkd(c,"Q232")
    KOSOVO = wkd(c,"Q1246")
    LATVIA = wkd(c,"Q211")
    LIECHTENSTEIN = wkd(c,"Q347")
    LITHUANIA = wkd(c,"Q37")
    LUXEMBOURG = wkd(c,"Q32")
    MACEDONIA = wkd(c,"Q221")
    MALTA = wkd(c,"Q233")
    MOLDOVA = wkd(c,"Q217")
    MONACO = wkd(c,"Q235")
    MONTENEGRO = wkd(c,"Q236")
    NETHERLANDS = wkd(c,"Q55")
    SOUTHERN_NETHERLANDS = wkd(c,"Q6581823")
    KINGDOM_OF_NETHERLANDS = wkd(c,"Q29999")
    NORWAY = wkd(c,"Q20")
    POLAND = wkd(c,"Q36")
    PORTUGAL = wkd(c,"Q45")
    ROMANIA = wkd(c,"Q218")
    RUSSIA = wkd(c,"Q159")
    SOVIET_UNION =wkd(c,"Q15180")
    RUSSIAN_EMPIRE = wkd(c,"Q34266")
    SAN_MARINO = wkd(c,"Q238")
    SERBIA = wkd(c,"Q403")
    YOUGOSLAVIA = wkd(c,"Q36704")
    SLOVAKIA = wkd(c,"Q214")
    SLOVENIA = wkd(c,"Q215")
    SPAIN = wkd(c,"Q29")
    KINGDOM_OF_CASTILLE = wkd(c,"Q179293")
    SWEDEN = wkd(c,"Q34")
    SWITZERLAND = wkd(c,"Q39")
    TURKEY = wkd(c,"Q43")
    OTTOMAN_EMPIRE = wkd(c,"Q12560")
    UKRAINE = wkd(c,"Q212")
    UNITED_KINGDOM = wkd(c,"Q145")
    UNITED_KINGDOM_OLD = wkd(c,"Q174193")
    KINGDOM_OF_ENGLAND = wkd(c,"Q179876")
    KINGDOM_OF_GREAT_BRITAIN = wkd(c,"Q161885")
    VATICAN_CITY = wkd(c,"Q237")


    in_albania = c.satisfy(TRAVERSIBLE_COUNTRY, [ALBANIA])
    in_andorra = c.satisfy(TRAVERSIBLE_COUNTRY, [ANDORRA])
    in_armenia = c.satisfy(TRAVERSIBLE_COUNTRY, [ARMENIA])

    in_austria_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [AUSTRIA, AUSTRIA_HUNGARY])
    is_austria_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Austrian people by occupation")], max_steps=5)
    in_austria = logical_ors([in_austria_wkd, is_austria_people])
    in_azerbaijan = c.satisfy(TRAVERSIBLE_COUNTRY, [AZERBAIJAN])
    in_belarus = c.satisfy(TRAVERSIBLE_COUNTRY, [BELARUS])
    in_belgium = c.satisfy(TRAVERSIBLE_COUNTRY, [BELGIUM])
    in_bosnia = c.satisfy(TRAVERSIBLE_COUNTRY, [BOSNIA])
    in_bulgaria = c.satisfy(TRAVERSIBLE_COUNTRY, [BULGARIA])
    in_croatia = c.satisfy(TRAVERSIBLE_COUNTRY, [CROATIA])
    in_cyprus = c.satisfy(TRAVERSIBLE_COUNTRY, [CYPRUS])
    in_czech_republic_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [CZECH_REPUBLIC,CZECHOSLOVAKIA])
    czhec_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Czechoslovak people")], max_steps=5)
    in_czech_republic = logical_ors([in_czech_republic_wkd, czhec_people])
    in_denmark_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [DENMARK])
    is_danish_legendary_figure = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Danish legendary figures")], max_steps=5)
    in_denmark = logical_ors([in_denmark_wkd,is_danish_legendary_figure])

    in_estonia = c.satisfy(TRAVERSIBLE_COUNTRY, [ESTONIA])
    in_finland = c.satisfy(TRAVERSIBLE_COUNTRY, [FINLAND])


    years_in_france = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in France")], max_steps=5)
    in_france_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [FRANCE])
    in_france = logical_ors([in_france_wkd,years_in_france])

    in_georgia = c.satisfy(TRAVERSIBLE_COUNTRY, [GEORGIA])

    years_in_germany = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in Germany")], max_steps=5)
    nazis =  c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Nazis")], max_steps=5)
    german_nobility = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:German nobility")], max_steps=7)
    in_germany_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [GERMANY, GERMANY_NAZI, GERMAN_EMPIRE, GERMAN_CONFEDERATION, EAST_GERMANY])
    in_germany = logical_ors([in_germany_wkd, years_in_germany, nazis, german_nobility])

    years_in_greece = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in Greece")], max_steps=5)
    ancient_greeks = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Ancient Greeks")], max_steps=7)
    greek_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Greek people by occupation")], max_steps=7)
    in_greece_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [GREECE])
    in_greece = logical_ors([in_greece_wkd,years_in_greece, ancient_greeks, greek_people])

    in_hungary = c.satisfy(TRAVERSIBLE_COUNTRY, [HUNGARY])
    in_iceland = c.satisfy(TRAVERSIBLE_COUNTRY, [ICELAND])
    in_ireland = c.satisfy(TRAVERSIBLE_COUNTRY, [IRELAND])
    in_italy_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [ITALY,NATIONAL_FASCIST_PARTY, KINGDOM_OF_ITALY, ROMAN_EMPIRE, ANCIENT_ROME])
    is_italian_politician = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Italian politicians")], max_steps=6)
    in_roman_empire = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Roman Empire")], max_steps=6)
    in_history_of_italy = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:History of Italy by region")], max_steps=6)
    italian_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Italian people by occupation")], max_steps=6)
    ancient_romans = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Ancient Romans")], max_steps=8)
    in_italy = logical_ors([in_italy_wkd, in_roman_empire, in_history_of_italy,
               is_italian_politician, italian_people, ancient_romans])
    in_kazakhstan = c.satisfy(TRAVERSIBLE_COUNTRY, [KAZAKHSTAN])
    in_kosovo = c.satisfy(TRAVERSIBLE_COUNTRY, [KOSOVO])
    in_latvia = c.satisfy(TRAVERSIBLE_COUNTRY, [LATVIA])
    in_liectenstein = c.satisfy(TRAVERSIBLE_COUNTRY, [LIECHTENSTEIN])
    in_lithuania =  c.satisfy(TRAVERSIBLE_COUNTRY, [LITHUANIA])
    in_luxembourg = c.satisfy(TRAVERSIBLE_COUNTRY, [LUXEMBOURG])
    in_macedonia = c.satisfy(TRAVERSIBLE_COUNTRY, [MACEDONIA])
    in_malta = c.satisfy(TRAVERSIBLE_COUNTRY, [MALTA])
    in_moldova = c.satisfy(TRAVERSIBLE_COUNTRY, [MOLDOVA])
    in_monaco = c.satisfy(TRAVERSIBLE_COUNTRY, [MONACO])
    in_montenegro = c.satisfy(TRAVERSIBLE_COUNTRY, [MONTENEGRO])
    in_netherlands_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [NETHERLANDS, KINGDOM_OF_NETHERLANDS, SOUTHERN_NETHERLANDS])
    dutch_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Dutch people by occupation")], max_steps=5)
    in_netherlands = logical_ors([in_netherlands_wkd, dutch_people])
    in_norway = c.satisfy(TRAVERSIBLE_COUNTRY, [NORWAY])
    in_poland = c.satisfy(TRAVERSIBLE_COUNTRY, [POLAND])
    in_portugal = c.satisfy(TRAVERSIBLE_COUNTRY, [PORTUGAL])
    in_romania = c.satisfy(TRAVERSIBLE_COUNTRY, [ROMANIA])
    russian_people = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Russian people by occupation")], max_steps=7)
    sport_in_the_soviet_union = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Sport in the Soviet Union")], max_steps=7)
    in_russia_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [RUSSIA, RUSSIAN_EMPIRE, SOVIET_UNION])
    in_russia = logical_ors([in_russia_wkd, russian_people, sport_in_the_soviet_union])
    in_san_marino = c.satisfy(TRAVERSIBLE_COUNTRY, [SAN_MARINO])
    in_serbia = c.satisfy(TRAVERSIBLE_COUNTRY, [SERBIA, YOUGOSLAVIA])
    in_slovakia = c.satisfy(TRAVERSIBLE_COUNTRY, [SLOVAKIA])
    in_slovenia = c.satisfy(TRAVERSIBLE_COUNTRY, [SLOVENIA])
    years_in_spain = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in Spain")], max_steps=5)
    in_spain_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [SPAIN, KINGDOM_OF_CASTILLE])
    in_spain = logical_ors([in_spain_wkd, years_in_spain])
    years_in_sweden = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in Sweden")], max_steps=5)
    in_sweden_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [SWEDEN])
    in_sweden = logical_ors([in_sweden_wkd, years_in_sweden])
    years_in_switzerland = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Years in Switzerland")], max_steps=5)
    in_switzerland_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [SWITZERLAND])
    in_switzerland = logical_ors([in_switzerland_wkd, years_in_switzerland ])
    in_turkey = c.satisfy(TRAVERSIBLE_COUNTRY, [TURKEY, OTTOMAN_EMPIRE])
    in_ukraine = c.satisfy(TRAVERSIBLE_COUNTRY, [UKRAINE])
    in_united_kingdom = c.satisfy(TRAVERSIBLE_COUNTRY,
                        [UNITED_KINGDOM, UNITED_KINGDOM_OLD, KINGDOM_OF_ENGLAND, KINGDOM_OF_GREAT_BRITAIN])
    popes = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Popes")], max_steps=5)
    in_vatican_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [VATICAN_CITY])
    in_vatican = logical_ors([popes, in_vatican_wkd])


    print("Artic and others")
    ARCTIC = wkd(c,"Q25322")
    INUIT = wkd(c,"Q189975")
    FAROE_ISLANDS = wkd(c,"Q4628")
    TONGA = wkd(c,"Q678")
    in_faroe_islands_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Faroe Islands")], max_steps=5)
    in_faroe_islands_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [FAROE_ISLANDS])
    in_faroe_islands = logical_ors([in_faroe_islands_wkp, in_faroe_islands_wkd])
    in_arctic = c.satisfy(TRAVERSIBLE_COUNTRY, [ARCTIC,INUIT])
    in_tonga_wkd = c.satisfy(TRAVERSIBLE_COUNTRY, [TONGA])
    in_tonga_wkp = c.satisfy([wprop.CATEGORY_LINK], [wkp(c, "Category:Tonga")], max_steps=5)
    in_tonga = logical_ors([in_tonga_wkd,in_tonga_wkp])



    is_unlocated = logical_ors([is_Wiki_Links,is_taxon])
    is_unlocated_not = logical_negate(is_unlocated,[is_populated_place,
                              is_in_outer_space_not_earth,in_tanzania])
    is_unlocated_only = logical_ors([is_unlocated_not,is_other])


    COUNTRIES = [ALGERIA, ANGOLA, BENIN, BOTSWANA, BURKINA_FASO, BURUNDI, CAPE_VERDE, CAMEROON, CHAD,
        CENTRAL_AFRICAN_REPUBLIC, COMOROS, DEMOCRATIC_REPUBLIC_OF_CONGO, REPUBLIC_OF_CONGO, DJIBOUTI,
        EGYPT, EQUATORIAL_GUINEA, ERITREA, ETHIOPIA, GABON, THE_GAMBIA, GHANA, GUINEA, GUINEA_BISSAU, IVORY_COAST,
        LESOTHO, KENYA, LIBERIA, LIBYA, MADAGASCAR, MALAWI, MALI, MAURITANIA,MAURITIUS, MOROCCO, MOZAMBIQUE,
        NAMIBIA, NIGER, NIGERIA, RWANDA,SAHARI_ARAB_DEOMOCRATIC_REPUBLIC, SAO_TOME_AND_PRINCIPE, SENEGAL,
        SEYCHELLES, SIERRA_LEONE, SOMALIA, SOMALI_LAND, SOUTH_AFRICA, SUDAN, TANZANIA, TOGO,
        TUNISIA, UGANDA, WESTERN_SAHARA, ZAMBIA, ZIMBABWE,
        AUSTRALIA, FIJI,INDONESIA,KIRIBATI, MARSHALL_ISLANDS,
        FEDERATED_STATES_OF_MICRONESIA, NAURU, NEW_ZEALAND, PAPUA_NEW_GUINEA, SAMOA, SOLOMON_ISLANDS, VANUATU,
        ARGENTINA, BOLIVIA, BRAZIL, CHILE, COLOMBIA, ECUADOR, GUYANA, PARAGUAY, PERU, SURINAME, TRINIDAD_AND_TOBAGO,
        URUGUAY, VENEZUELA,
        BELIZE, COSTA_RICA,EL_SALVADOR, GUATEMALA, HONDURAS, NICARAGUA, PANAMA,
        ANTIGUA_BARBUDA, BAHAMAS, BARBADOS, CANADA, CUBA, DOMINICAN_REPUBLIC, GRENADA, GUATEMALA, HAITI, JAMAICA, MEXICO,
        SAINT_KITTS_AND_NEVIS, SAINT_LUCIA, SAINT_VINCENT_AND_GRENADINES, UNITED_STATES,
        ALBANIA, ANDORRA, ARMENIA, AUSTRIA, AUSTRIA_HUNGARY, AZERBAIJAN, BELARUS, BELGIUM, BOSNIA, BULGARIA, CROATIA,
        CYPRUS,
        CZECH_REPUBLIC, CZECHOSLOVAKIA,
        DENMARK,  ESTONIA, FINLAND, FRANCE, GEORGIA, GERMANY, GERMANY_NAZI, GREECE, HUNGARY, ICELAND,
        IRELAND, ITALY, NATIONAL_FASCIST_PARTY, KINGDOM_OF_ITALY, ROMAN_EMPIRE,
        KAZAKHSTAN, KOSOVO, LATVIA, LIECHTENSTEIN, LITHUANIA, LUXEMBOURG, MACEDONIA, MALTA,
        MOLDOVA, MONACO, MONTENEGRO, NORWAY,
        NETHERLANDS, KINGDOM_OF_NETHERLANDS, SOUTHERN_NETHERLANDS,
        POLAND, PORTUGAL, ROMANIA,
        RUSSIA, RUSSIAN_EMPIRE, SOVIET_UNION,
        SAN_MARINO,
        SERBIA, YOUGOSLAVIA,
        SLOVAKIA,
        SLOVENIA, SPAIN, SWEDEN, SWITZERLAND,
        TURKEY, OTTOMAN_EMPIRE, UKRAINE,
        UNITED_KINGDOM, UNITED_KINGDOM_OLD, KINGDOM_OF_ENGLAND, KINGDOM_OF_GREAT_BRITAIN,
        AFGHANISTAN, BANGLADESH, BRUNEI, CAMBODIA, CHINA, CYPRUS, EAST_TIMOR, EGYPT, GEORGIA, INDIA, INDONESIA,
        IRAN, IRAQ, ISRAEL, JAPAN, KAZAKHSTAN, KUWAIT, KYRGYZSTAN, LAOS, LEBANON, MALAYSIA, MALDIVES, MONGOLIA,
        MYANMAR, NEPAL, NORTH_KOREA, OMAN, PALESTINE, PAKISTAN, PHILIPPINES, QATAR, SAUDI_ARABIA, SINGAPORE, SOUTH_KOREA, SRI_LANKA,
        SYRIA, TAJIKISTAN, TAIWAN, THAILAND, TURKMENISTAN, UNITED_ARAB_EMIRATES, UZBEKISTAN, VIETNAM, YEMEN,
        VATICAN_CITY,
        ARCTIC, FAROE_ISLANDS, TONGA
        ]



    located_somewhere_wkd = c.satisfy([wprop.COUNTRY_OF_CITIZENSHIP, wprop.COUNTRY], COUNTRIES)
    located_somewhere = logical_ors([ located_somewhere_wkd, in_austria, in_afghanistan, in_china, in_france,
    in_sweden, in_china, in_switzerland, in_germany, years_in_usa, in_greece,
    in_south_korea, in_italy,
    in_denmark, in_spain, in_iraq, in_egypt, in_vatican, in_canada,
    in_faroe_islands, in_netherlands, in_russia, in_samoa, in_syria, in_tonga, in_zambia ])

    is_unlocated_politician =  logical_negate(is_politician,[located_somewhere])

    is_unlocated_painter = logical_negate(is_painter, [located_somewhere])

    is_unlocated_association_football_player = logical_negate(is_association_football_player, [located_somewhere])


    return {
        "Algeria": in_algeria,
        "Angola":  in_angola,
        "Benin": in_benin,
        "BOSTWANA": in_botswana,
        "BURKINA_FASO": in_burkina_faso,
        "BURUNDI": in_burundi,
        "CAPE_VERDE": in_cape_verde,
        "CAMEROON": in_cameroon,
        "CHAD": in_chad,
        "CENTRAL AFRICAN REPUBLIC": in_central_african_republic,
        "COMOROS": in_comoros,
        "DEMOCRATIC_REPUBLIC_OF_CONGO": in_democratic_republic_congo,
        "REPUBLIC_OF_CONGO": in_republic_of_congo,
        "DJIBOUTI": in_djibouti,
        "EGYPT": in_egypt,
        "EQUATORIAL_GUINEA":  in_equatorial_guinea,
        "ERITREA": in_eritrea,
        "ETHIOPIA": in_ethiopia,
        "GABON": in_gabon,
        "THE_GAMBIA": in_the_gambia,
        "GHANA": in_ghana,
        "GUINEA": in_guinea,
        "GUINEA_BISSAU": in_guinea_bissau,
        "IVORY_COAST": in_ivory_coast,
        "LESOTHO": in_lesotho,
        "KENYA": in_kenya,
        "LIBERIA": in_liberia,
        "LIBYA": in_libya,
        "Madagascar": in_madagascar,
        "Malawi":  in_malawi,
        "Mali": in_mali,
        "Mauritania": in_mauritania,
        "Mauritius": in_mauritius,
        "Morocco": in_morrocco,
        "Mozambique": in_mozambique,
        "Namibia": in_namibia,
        "Niger": in_niger,
        "Nigeria": in_nigeria,
        "Rwanda": in_rwanda,
        "Sahrawi_Arab_Democratic_Republic": in_sadr,
        "Sao_Tome_and_Principe": in_stap,
        "Senegal": in_senegal,
        "Seychelles": in_seychelles,
        "Sierra_Leone": in_sierra_leone,
        "Somalia": in_somalia,
        "Somaliland‎": in_somali_land,
        "South_Africa‎": in_south_africa,
        "South_Sudan‎": in_ssudan,
        "Sudan": in_sudan,
        "SWAZILAND": in_swaziland,
        "TANZANIA": in_tanzania,
        "TOGO": in_togo,
        "TUNISIA": in_tunisia,
        "Uganda": in_uganda,
        "Western Sahara": in_western_sahara,
        "Zambia": in_zambia,
        "Zimbabwe": in_zimbabwe,


        "AUSTRALIA": in_australia,
        "FIJI": in_fiji,
        "INDONESIA": in_indonesia,
        "KIRIBATI": in_kiribati,
        "MARSHALL_ISLANDS": in_marshall_islands,
        "FEDERATED_STATES_OF_MICRONESIA": in_federates_states_of_micronesia,
        "NAURU": in_nauru,
        "NEW_ZEALAND": in_new_zealand,
        "PAPUA_NEW_GUINEA": in_papua_new_guinea,
        "SAMOA": in_samoa,
        "SOLOMON_ISLANDS": in_solomon_islands,
        "VANUATU": in_vanuatu,


        "ARGENTINA": in_argentina,
        "BOLIVIA": in_bolivia,
        "BRAZIL": in_brazil,
        "CHILE": in_chile,
        "COLOMBIA": in_colombia,
        "ECUADOR": in_ecuador,
        "GUYANA": in_guyana,
        "PARAGUAY": in_paraguay,
        "PERU": in_peru,
        "SURINAME": in_suriname,
        "TRINIDAD_AND_TOBAGO": in_trinidad_and_tobago,
        "URUGUAY": in_uruguay,
        "VENEZUELA": in_venezuela,


        "BELIZE": in_belize,
        "COSTA_RICA": in_costa_rica,
        "EL_SALVADOR": in_el_salvador,
        "GUATEMALA": in_guatemala,
        "HONDURAS": in_honduras,
        "NICARAGUA": in_nicaragua,
        "PANAMA": in_panama,


        "ANTIGUA_BARBUDA": in_antigua_barbuda,
        "BAHAMAS": in_bahamas,
        "BARBADOS": in_barbados,
        "CANADA": in_canada,
        "CUBA": in_cuba,
        "DOMINICAN REPUBLIC": in_dominican_republic,
        "GRENADA": in_grenada,
        "GUATEMALA": in_guatemala,
        "HAITI": in_haiti,
        "JAMAICA": in_jamaica,
        "MEXICO": in_mexico,
        "SAINT_KITTS_AND_NEVIS": in_Saint_Kitts_and_Nevis,
        "SAINT_LUCIA": in_saint_lucia,
        "SAINT_VINCENT_AND_GRENADINES": in_saint_vincent_and_grenadines,
        "UNITED_STATES": in_united_states,


        "ALBANIA": in_albania,
        "ANDORRA": in_andorra,
        "ARMENIA": in_armenia,
        "AUSTRIA": in_austria,
        "AZERBAIJAN": in_azerbaijan,
        "BELARUS": in_belarus,
        "BELGIUM": in_belgium,
        "BOSNIA": in_bosnia,
        "BULGARIA": in_bulgaria,
        "CROATIA": in_croatia,
        "CYPRUS": in_cyprus,
        "CZECH REPUBLIC": in_czech_republic,
        "DENMARK": in_denmark,
        "ESTONIA": in_estonia,
        "FINLAND": in_finland,
        "FRANCE": in_france,
        "GEORGIA": in_georgia,
        "GERMANY": in_germany,
        "GREECE": in_greece,
        "HUNGARY": in_hungary,
        "ICELAND": in_iceland,
        "IRELAND": in_ireland,
        "ITALY": in_italy,
        "KAZAKHSTAN": in_kazakhstan,
        "KOSOVO": in_kosovo,
        "LATVIA": in_latvia,
        "LIECHTENSTEIN": in_liectenstein,
        "LITHUANIA": in_lithuania,
        "LUXEMBOURG": in_luxembourg,
        "MACEDONIA": in_macedonia,
        "MALTA": in_malta,
        "MOLDOVA": in_moldova,
        "MONACO": in_monaco,
        "MONTENEGRO": in_montenegro,
        "NORWAY": in_norway,
        "NETHERLANDS": in_netherlands,
        "POLAND": in_poland,
        "PORTUGAL": in_portugal,
        "ROMANIA": in_romania,
        "RUSSIA": in_russia,
        "SAN MARINO": in_san_marino,
        "SERBIA": in_serbia,
        "SLOVAKIA": in_slovakia,
        "SLOVENIA": in_slovenia,
        "SPAIN": in_spain,
        "SWEDEN": in_sweden,
        "SWITZERLAND": in_switzerland,
        "TURKEY": in_turkey,
        "UKRAINE": in_ukraine,
        "UNITED KINGDOM": in_united_kingdom,


        "AFGHANISTAN": in_afghanistan,
        "BANGLADESH": in_bangladesh,
        "BHUTAN": in_bhutan,
        "BRUNEI": in_brunei,
        "CAMBODIA": in_cambodia,
        "CHINA": in_china,
        "CYPRUS": in_cyprus,
        "EAST TIMOR": in_east_timor,
        "EGYPT": in_egypt,
        "GEORGIA": in_georgia,
        "INDIA": in_india,
        "INDONESIA": in_indonesia,
        "IRAN": in_iran,
        "IRAQ": in_iraq,
        "ISRAEL": in_israel,
        "JAPAN": in_japan,
        "JORDAN": in_jordan,
        "KAZAKHSTAN": in_kazakhstan,
        "KUWAIT": in_kuwait,
        "KYRGYZSTAN": in_kyrgyzstan,
        "LAOS": in_laos,
        "LEBANON": in_lebanon,
        "MALAYSIA": in_malaysia,
        "MALDIVES": in_maldives,
        "MONGOLIA": in_mongolia,
        "MYANMAR": in_myanmar,
        "NEPAL": in_nepal,
        "NORTH_KOREA": in_north_korea,
        "OMAN": in_oman,
        "PALESTINE": in_palestine,
        "PAKISTAN": in_pakistan,
        "PHILIPPINES": in_philippines,
        "QATAR": in_qatar,
        "SAUDI_ARABIA": in_saudi_arabia,
        "SINGAPORE": in_singapore,
        "SOUTH_KOREA": in_south_korea,
        "SRI LANKA": in_sri_lanka,
        "SYRIA": in_syria,
        "TAJIKISTAN": in_tajikistan,
        "TAIWAN": in_taiwan,
        "THAILAND": in_thailand,
        "TURKMENISTAN": in_turkmenistan,
        "UNITED_ARAB_EMIRATES": in_united_arab_emirates,
        "UZBEKISTAN": in_uzbekistan,
        "VIETNAM": in_vietnam,
        "YEMEN": in_yemen,
        "OUTERSPACE": is_in_outer_space_not_earth,

        "ARCTIC": in_arctic,
        "FAROE_ISLANDS": in_faroe_islands,
        "TONGA": in_tonga,

        "UNLOCATED": is_unlocated_only,
        "USA_ROADS": in_usa_roads,
        "POLITICIAN": is_politician,
        "UNLOCATED_POLITICIAN": is_unlocated_politician,
        "UNLOCATED_PAINTER": is_unlocated_painter,
        "UNLOCATED_ASSOCIATION_FOOTBALL_PLAYER": is_unlocated_association_football_player
    }
