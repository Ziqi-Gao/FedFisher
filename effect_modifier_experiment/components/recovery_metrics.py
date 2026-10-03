"""Legacy binary ranking arithmetic; callers retain their own contrast scales."""
import math
import numpy as np
from utils.prediction_intervention import evaluate_signal_recovery


def signal_recovery_with_ties(importance, signal_dim, metric):
    """Primary Top-k recovery integrates uniform ties at the selection boundary.

    The legacy stable index tie break is preserved only as a named diagnostic;
    it cannot reward X1/X2 simply for appearing first in an all-zero ranking.
    Exact equality of observed importance values defines a tie.
    """
    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("recovery requires finite nonnegative feature importances")
    if type(signal_dim) is not int or not 1 <= signal_dim <= values.size:
        raise ValueError("invalid recovery signal count")
    diagnostic = evaluate_signal_recovery(
        values, signal_dim, metric, signal_indices=list(range(signal_dim)),
        candidate_indices=list(range(values.size)))
    diagnostic = {key: value if math.isfinite(value) else None for key, value in diagnostic.items()}
    cutoff = np.sort(values)[-signal_dim]
    above = values > cutoff
    tied = values == cutoff
    slots = signal_dim - int(above.sum())
    tied_count = int(tied.sum())
    true_above = int(above[:signal_dim].sum())
    true_tied = int(tied[:signal_dim].sum())
    expected_hits = true_above + slots * true_tied / tied_count
    if true_above + true_tied != signal_dim or true_tied > slots:
        all_selected = 0.0
    else:
        all_selected = math.comb(tied_count - true_tied, slots - true_tied) / math.comb(tied_count, slots)
    return {
        "expected_topk_hits": expected_hits,
        "expected_topk_precision": expected_hits / signal_dim,
        "probability_all_signals_selected": all_selected,
        "auroc": diagnostic["auroc"],
        "boundary_tie_size": tied_count,
        "boundary_slots": slots,
        "tie_rule": "uniform_without_replacement_within_exact_score_boundary_ties",
        "deterministic_index_tiebreak_diagnostic": diagnostic,
    }
