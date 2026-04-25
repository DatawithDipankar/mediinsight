"""
mock_data.py
------------
Generates realistic mock PubMed articles on intermittent fasting for
offline / sandbox testing. Replace with real PubMedRetriever calls
when running in an environment with access to eutils.ncbi.nlm.nih.gov.
"""

import random

JOURNALS = [
    "New England Journal of Medicine",
    "JAMA Internal Medicine",
    "The Lancet",
    "Diabetes Care",
    "Obesity Reviews",
    "Cell Metabolism",
    "Nutrients",
    "American Journal of Clinical Nutrition",
    "International Journal of Obesity",
    "Metabolism",
]

AUTHORS_POOL = [
    "Krista Varady", "Mark Mattson", "Valter Longo", "Jason Fung",
    "Satchidananda Panda", "Rafael de Cabo", "Courtney Peterson",
    "Michelle Harvie", "Leanne Redman", "Frank Madeo",
    "Eric Ravussin", "Donald Layman", "David Ludwig", "Ethan Weiss",
]

IF_PROTOCOLS = ["16:8", "5:2", "alternate-day fasting", "one-meal-a-day (OMAD)", "time-restricted eating"]
CONDITIONS   = ["obesity", "type 2 diabetes", "metabolic syndrome", "insulin resistance", "dyslipidemia"]
OUTCOMES     = [
    "significant reductions in body weight",
    "improved fasting insulin levels",
    "reduced HbA1c",
    "decreased visceral fat",
    "improved lipid profiles",
    "enhanced insulin sensitivity",
    "reduced inflammatory markers",
    "improved blood pressure",
    "decreased fasting glucose",
    "improved beta-cell function",
]

TITLE_TEMPLATES = [
    "Effects of {protocol} intermittent fasting on {condition}: a randomized controlled trial",
    "{protocol} time-restricted eating and {condition}: systematic review and meta-analysis",
    "Intermittent fasting ({protocol}) improves {outcome} in patients with {condition}",
    "Comparative study of {protocol} fasting versus caloric restriction in {condition}",
    "Short-term {protocol} intermittent fasting: impact on metabolic markers in {condition}",
    "A pilot study of {protocol} fasting protocol in adults with {condition}",
    "Mechanistic insights into {protocol} fasting and {outcome}",
    "Long-term adherence to {protocol} fasting in {condition}: a cohort study",
    "Gut microbiome changes during {protocol} fasting in {condition} patients",
    "Circadian alignment and {protocol} time-restricted eating in {condition}",
]


def _make_article(i: int) -> dict:
    protocol  = random.choice(IF_PROTOCOLS)
    condition = random.choice(CONDITIONS)
    outcome   = random.choice(OUTCOMES)
    title     = random.choice(TITLE_TEMPLATES).format(
        protocol=protocol, condition=condition, outcome=outcome
    )
    n_authors = random.randint(2, 5)
    authors   = ", ".join(random.sample(AUTHORS_POOL, n_authors))
    year      = str(random.randint(2015, 2024))
    pmid      = str(30000000 + i)

    abstract = {
        "BACKGROUND": (
            f"Intermittent fasting (IF) using the {protocol} protocol has emerged as a promising "
            f"non-pharmacological strategy for managing {condition}. Despite growing interest, "
            "the metabolic effects and long-term adherence remain insufficiently characterised."
        ),
        "METHODS": (
            f"A total of {random.randint(40, 250)} adults with {condition} were enrolled in this "
            f"{random.choice(['12-week', '16-week', '24-week', '6-month'])} "
            f"{random.choice(['randomized controlled trial', 'prospective cohort study', 'crossover study'])}. "
            f"Participants followed the {protocol} IF protocol versus an isocaloric control diet. "
            "Primary outcomes included anthropometric measures and fasting metabolic markers."
        ),
        "RESULTS": (
            f"The IF group demonstrated {outcome} compared with controls "
            f"(p < {random.choice(['0.001', '0.01', '0.05'])}). "
            f"Body weight decreased by {random.uniform(2.0, 8.5):.1f} kg "
            f"and fasting glucose fell by {random.uniform(5, 25):.1f} mg/dL. "
            "No serious adverse events were reported."
        ),
        "CONCLUSIONS": (
            f"The {protocol} IF protocol is effective and well-tolerated in individuals with {condition}, "
            f"producing clinically meaningful {outcome}. Larger trials are warranted to confirm long-term safety."
        ),
    }

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": random.choice(JOURNALS),
        "authors": authors,
        "publication_date": year,
    }


def generate_mock_articles(n: int = 300) -> list[dict]:
    random.seed(42)
    return [_make_article(i) for i in range(n)]


def generate_mock_pmids(n: int = 300) -> list[str]:
    return [str(30000000 + i) for i in range(n)]
