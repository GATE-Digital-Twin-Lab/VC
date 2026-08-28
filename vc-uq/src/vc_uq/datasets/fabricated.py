"""Fabricated-entity questions: ``p_q = 0`` by construction.

These ask about entities that do not exist, so no correct answer exists and the
true per-sample correctness probability is exactly zero -- not estimated, known.
That makes them the only place in the study where the estimand itself is
observable, which the protocol leans on twice:

  * 6.4-6.6 -- a guaranteed non-empty ``U`` with known membership, and the
    cleanest population of the low-diversity-``U`` cell (a confident, stable,
    fabricated belief).
  * 8.3 -- unanswerable questions as an independent handle on that ``U``
    analysis, since membership is known rather than inferred.

Generated rather than downloaded so that ``p_q = 0`` is guaranteed: any harvested
"unanswerable" set risks containing items the model can in fact answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Invented proper nouns. Checked for implausibility as real entities: the point
# is that they are well-formed and unfamiliar, not that they are nonsense.
_SURNAMES = ["Vantreau", "Okonkwe-Blyth", "Marchetti-Sorn", "Delacroix-Venn",
             "Hallowmere", "Strindqvist", "Ashgrove-Petit", "Belanova",
             "Corvellier", "Draymoor", "Eskilsen-Vray", "Fontanelle-Crisp"]
_FORENAMES = ["Aurelio", "Bettina", "Casimir", "Delphine", "Emeric", "Fiora",
              "Gustavo", "Halina", "Ivo", "Juniper", "Konstantin", "Liesl"]
_TITLES = ["The Crimson Aviary", "Ledgers of Ash", "The Pemberton Variation",
           "Nine Doors to Vasilburg", "A Cartography of Silence",
           "The Hollow Meridian", "Saltmarsh Interlude", "The Vellum Engine",
           "Quintessence of Dross", "Twelve Nights in Aldouri"]
_PLACES = ["Zubrowka", "Marisol Province", "the Kessel Basin", "Vantorra",
           "the Republic of Ostrelia", "Nyhavnsted", "the Calloway Reach",
           "Uppermarch", "the Sableholm Isles", "Terranova Minor"]
_FIELDS = ["marine bioacoustics", "computational paleography",
           "lattice thermodynamics", "forensic musicology",
           "glacial microbiology", "syntactic archaeology"]
_INSTITUTIONS = ["the Kalvenhoff Institute", "Ravensmoor College",
                 "the Dunhollow Observatory", "the Institute of Applied Vantics",
                 "Bergstrand Polytechnic"]

_TEMPLATES: list[tuple[str, str]] = [
    ("Who directed the {year} film {title}?", "title_year"),
    ("In what year was {forename} {surname} awarded the Kalvenhoff Prize for {field}?",
     "person_field"),
    ("What is the capital city of {place}?", "place"),
    ("Which novel by {forename} {surname} won the Ostrelian Book Award?", "person"),
    ("How many permanent staff does {institution} employ?", "institution"),
    ("What is the population of {place} as of the most recent census?", "place"),
    ("Who founded {institution}, and in what year?", "institution"),
    ("What was {forename} {surname}'s contribution to {field}?", "person_field"),
    ("On what date did the {title} incident occur in {place}?", "title_place"),
    ("Which element was first isolated by {forename} {surname}?", "person"),
]

#: There is no correct answer, so the reference is the correct *behaviour*.
A_STAR = "No such entity exists; the question has no correct answer."


def build(n_questions: int, seed: int = 991) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < n_questions and attempts < n_questions * 50:
        attempts += 1
        template, _kind = _TEMPLATES[int(rng.integers(len(_TEMPLATES)))]
        text = template.format(
            year=int(rng.integers(1954, 2011)),
            title=_TITLES[int(rng.integers(len(_TITLES)))],
            forename=_FORENAMES[int(rng.integers(len(_FORENAMES)))],
            surname=_SURNAMES[int(rng.integers(len(_SURNAMES)))],
            place=_PLACES[int(rng.integers(len(_PLACES)))],
            field=_FIELDS[int(rng.integers(len(_FIELDS)))],
            institution=_INSTITUTIONS[int(rng.integers(len(_INSTITUTIONS)))],
        )
        if text in seen:
            continue
        seen.add(text)
        rows.append({
            "q_id": f"fab_{len(rows):05d}",
            "dataset": "fabricated",
            "question": text,
            "a_star": A_STAR,
            "p_q_true": 0.0,     # by construction, not by measurement
        })
    if len(rows) < n_questions:
        raise ValueError(
            f"template pool exhausted at {len(rows)} unique questions; "
            f"add templates or entities to reach {n_questions}"
        )
    return pd.DataFrame(rows)
