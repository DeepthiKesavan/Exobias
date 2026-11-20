"""ExoBias - Exoplanet Uncertainty Analysis System (entry page: Project Overview).

Run with:  streamlit run ExoBias.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.app_helpers import (architecture_strip, card, dataset_guide, experiment_cards, gallery, hero, init_state, note,
                             results_showcase, setup_page, stat_row, steps_flow)

setup_page("Project Overview")
init_state()

hero(
    title="ExoBias",
    subtitle="Exoplanet Uncertainty Analysis System",
    text=("An uncertainty-aware machine learning framework that measures how different treatments of uncertain "
          "<i>Candidate</i> labels change classifier reliability and the population patterns read from NASA's Kepler "
          "Objects of Interest catalogue."),
    meta="Data source: NASA Exoplanet Archive &nbsp;·&nbsp; Kepler DR25 KOI catalogue",
)

stat_row([
    ("3", "Label-handling strategies"),
    ("5", "Model families"),
    ("kepid", "Host-star grouped splits"),
    ("70", "Automated tests"),
])

st.markdown("### The Central Inquiry")
note("<b>How do different strategies for handling uncertain candidate labels affect machine learning reliability "
     "and the interpretation of exoplanet population patterns?</b>")

c1, c2 = st.columns(2)
with c1:
    card(
        "Research problem",
        "In the Kepler catalogue a <b>Candidate</b> is neither a confirmed planet nor a false positive. Keeping candidates as they "
        "are, filtering them by vetting confidence, or down-weighting them may change both a classifier's reliability and the "
        "population patterns we infer. ExoBias measures that effect instead of assuming it.",
    )
with c2:
    card(
        "Dataset",
        "Kepler <b>DR25 KOI</b> table from the NASA Exoplanet Archive. DR25 mainly contains <b>CANDIDATE</b> and "
        "<b>FALSE POSITIVE</b> dispositions; <b>CONFIRMED</b> may be absent. The application detects the classes present, never "
        "fabricates data, and enables the three-class experiment only when all three classes genuinely exist.",
    )

st.markdown("### Worlds Beyond Our Sun")
gallery([("gas_giant.jpg", "Gas giant: the large, easily detected planets that dominate transit surveys"),
         ("ocean_world.jpg", "Ocean world: a small planet, far harder for Kepler to detect"),
         ("ice_world.jpg", "Ice world: cold, rocky planets sit at the edge of detectability")])
gallery([("transit.jpg", "A transit: a planet crossing its star dims the light, and that dip is how Kepler finds candidates"),
         ("nebula.jpg", "Stellar nursery: the kind of region where new stars, and eventually planetary systems, form")])
st.caption("Illustrative renders generated for this project (scripts/make_space_images.py). They are artistic impressions, not telescope photographs.")

st.markdown("### Results at a Glance")
results_showcase()

st.markdown("### Get the dataset")
dataset_guide(compact=True)

st.markdown("### Workflow")
steps_flow([
    ("Ingest and validate", "Upload the KOI CSV. Classes, columns and leakage risks are checked before anything trains."),
    ("Clean and split", "Row counts reported per stage. Train/validation/test split grouped by host star (kepid)."),
    ("Train ANN and baselines", "Experiments A, B and C, each with an ANN, Logistic Regression, Random Forest and optional models."),
    ("Evaluate reliability", "Accuracy, macro-F1, log loss, Brier score, ECE, ROC and precision-recall on one common test set."),
    ("Analyse populations", "Radius, period and host-star patterns under each strategy, with bootstrap intervals and effect sizes."),
    ("Test robustness and report", "Seeds, thresholds, weights, split types and features; export CSV, graphs and a summary."),
])

st.markdown("### The three experiments")
experiment_cards([
    dict(letter="A", title="Baseline ANN", color="#3987e5",
         body="All eligible Candidate and False Positive records. No vetting score is used. This is the reference model for B and C.",
         formula="train on every eligible row"),
    dict(letter="B", title="High-confidence filtered ANN", color="#d95926",
         body="Keep Candidates with a high score and False Positives with a low score. Trained on a smaller population, evaluated on the same original test set.",
         formula="Candidate: score ≥ t &nbsp;|&nbsp; False Positive: score ≤ 1 − t"),
    dict(letter="C", title="Confidence-weighted ANN", color="#199e70",
         body="Every record is kept, but its training weight follows the catalogue score. Weights are used in training only; test metrics are unweighted.",
         formula="weight = score (Candidate), 1 − score (False Positive)"),
])

st.markdown("### ANN architecture")
architecture_strip(["Input · selected features", "Dense 64 · ReLU", "Dropout 0.20", "Dense 32 · ReLU", "Softmax · K classes"])
st.markdown(
    "Adam optimiser (learning rate 0.001), sparse categorical cross-entropy, batch size 32, up to 100 epochs with early stopping "
    "(patience 12 on validation loss, best weights restored) and fixed random seeds. Every layer is configurable in the ANN Workbench. "
    "The ANN is compared against traditional models; it is **not assumed to win**."
)
