"""Round-2 evaluation metrics (scripts/r2_common.py): per-secret recall, precision targets, alarm budgets."""
import numpy as np

from scripts.r2_common import recall_at_precision, record_scores, threshold_for_budget


def test_record_scores_count_only_overlapping_candidates_for_secrets():
    # record 0: secret, one overlapping candidate (0.9) and one unrelated candidate (0.99)
    # record 1: non-secret, candidates 0.2 and 0.7 -> its score is the max, 0.7
    # record 2: secret caught by a provider regex; record 3: secret with no candidate (the ceiling)
    # record 4: non-secret flagged by a provider regex -> a false alarm at any threshold
    d = {"y": np.array([1, 0, 0, 0]), "rec": np.array([0, 0, 1, 1]), "label": np.array([1, 0, 1, 1, 0]),
         "regex": np.array([False, False, True, False, True])}
    s = record_scores(d, np.array([0.9, 0.99, 0.2, 0.7]))
    assert s.tolist() == [0.9, 0.7, 1.0, 0.0, 1.0]


def test_recall_at_precision_picks_the_highest_recall_meeting_the_target():
    label = np.array([1, 1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    assert recall_at_precision(label, scores, 1.0) == (2 / 3, 0.8)
    assert recall_at_precision(label, scores, 0.75) == (1.0, 0.6)


def test_recall_at_precision_never_cuts_inside_a_tie():
    # Taking only the first 1.0 would claim precision 1.0, but threshold 1.0 admits all three.
    label = np.array([1, 0, 0, 1])
    scores = np.array([1.0, 1.0, 1.0, 0.5])
    assert recall_at_precision(label, scores, 0.9) == (0.0, 1.0)


def test_threshold_for_budget_allows_exactly_the_budgeted_alarms():
    d = {"n_files": np.array(1000)}
    p = np.array([0.95, 0.9, 0.5, 0.4, 0.1])
    t = threshold_for_budget(d, p, per_1k=2)  # 2 alarms allowed in 1,000 files
    assert (p >= t).sum() == 2
