import json
import time
import re
import argparse

from wikidata_linker_utils.wikipedia import iterate_articles

from multiprocessing import Pool

CATEGORY_PREFIXES = [
    "Category:",
    "Catégorie:",
    "Categorie:",
    "Categoría:",
    "Categoria:",
    "Kategorie:",
    "Kategoria:",
    "Категория:",
    "Kategori:"
]

category_link_pattern = re.compile(
    r"\[\[((?:" + "|".join(CATEGORY_PREFIXES) + r")[^\]\[]*)\]\]"
)
redirection_link_pattern = re.compile(r"(?:#REDIRECT|#weiterleitung|#REDIRECCIÓN|REDIRECIONAMENTO)\s*\[\[([^\]\[]*)\]\]", re.IGNORECASE)
anchor_link_pattern = re.compile(r"\[\[([^\]\[:]*)\]\]")


def category_link_job(args):
    """
    Performing map-processing on different articles
    (in this case, just remove internal links)
    """
    article_name, lines = args
    found_tags = []
    for match in re.finditer(category_link_pattern, lines):
        match_string = match.group(1).strip()
        if "|" in match_string:
            link, _ = match_string.rsplit("|", 1)
            link = link.strip().split("#")[0]
        else:
            link = match_string

        if len(link) > 0:
            found_tags.append(link)
    return (article_name, found_tags)

def redirection_link_job(args):
    """
    Performing map-processing on different articles
    (in this case, just remove internal links)
    """
    article_name, lines = args
    found_tags = []
    for match in re.finditer(redirection_link_pattern, lines):
        if match is None:
            continue
        if match.group(1) is None:
            continue
        match_string = match.group(1).strip()
        if "|" in match_string:
            link, _ = match_string.rsplit("|", 1)
            link = link.strip().split("#")[0]
        else:
            link = match_string

        if len(link) > 0:
            found_tags.append(link)
    return (article_name, found_tags)


def anchor_finding_job(args):
    """
    Performing map-processing on different articles
    (in this case, just remove internal links)
    """
    article_name, lines = args
    found_tags = []
    for match in re.finditer(anchor_link_pattern, lines):
        match_string = match.group(1).strip()

        if "|" in match_string:
            link, anchor = match_string.rsplit("|", 1)
            link = link.strip().split("#")[0]
            anchor = anchor.strip()
        else:
            anchor = match_string
            link = match_string

        if len(anchor) > 0 and len(link) > 0:
            found_tags.append((anchor, link))
    return (article_name, found_tags)



def anchor_category_redirection_link_job(args):
    article_name, found_redirections = redirection_link_job(args)
    article_name, found_categories = category_link_job(args)
    article_name, found_anchors = anchor_finding_job(args)
    return (article_name, (found_anchors, found_redirections, found_categories))


def run_jobs(worker_pool, pool_jobs, outfile_anchors, outfile_redirections, outfile_category_links):
    results = worker_pool.map(anchor_category_redirection_link_job, pool_jobs)
    for article_name, result in results:
        anchor_links, redirect_links, category_links = result
        for link in redirect_links:
            outfile_redirections.write(article_name + "\t" + link + "\n")
        for link in category_links:
            outfile_category_links.write(article_name + "\t" + link + "\n")
        if ":" not in article_name:
            outfile_anchors.write(article_name + "\t" + article_name + "\t" + article_name + "\n")
            for anchor, link in anchor_links:
                outfile_anchors.write(article_name + "\t" + anchor + "\t" + link + "\n")


def parse_wiki(path,
               anchors_path,
               redirections_path,
               category_links_path,
               threads=1,
               max_jobs=10):
    t0 = time.time()
    jobs = []
    pool = Pool(processes=threads)
    try:
        with open(redirections_path, "wt") as fout_redirections, open(category_links_path, "wt") as fout_category_links, open(anchors_path, "wt") as fout_anchors:
            for article_name, lines in iterate_articles(path):
                jobs.append((article_name, lines))
                if len(jobs) >= max_jobs:
                    run_jobs(pool, jobs, fout_anchors, fout_redirections, fout_category_links)
                    jobs = []
            if len(jobs) > 0:
                run_jobs(pool, jobs, fout_anchors, fout_redirections, fout_category_links)
                jobs = []
    finally:
        pool.close()
    t1 = time.time()
    print("%.3fs elapsed." % (t1 - t0,))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("wiki",
        help="Wikipedia dump file (xml).")
    parser.add_argument("out_anchors",
        help="File where anchor information should be saved (tsv).")
    parser.add_argument("out_redirections",
        help="File where redirection information should be saved (tsv).")
    parser.add_argument("out_category_links",
        help="File where category link information should be saved (tsv).")

    def add_int_arg(name, default):
        parser.add_argument("--%s" % (name,), type=int, default=default)

    add_int_arg("threads", 8)
    add_int_arg("max_jobs", 10000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    parse_wiki(
        path=args.wiki,
        anchors_path=args.out_anchors,
        redirections_path=args.out_redirections,
        category_links_path=args.out_category_links,
        threads=args.threads,
        max_jobs=args.max_jobs
    )

if __name__ == "__main__":
    main()
