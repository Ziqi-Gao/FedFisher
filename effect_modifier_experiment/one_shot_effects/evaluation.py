"""Simulation-only references; no evaluator object is accepted by a server."""
from itertools import combinations
import math

import numpy as np

from ..causal_comparison.evaluation import signal_recovery_with_ties, array_hash
from .interventions import make_permutations, perturbed_covariates, permutation_hash


def predictions(model, x):
    value = np.asarray(model.predict_tau(x),dtype=np.float64)
    if value.shape!=(len(x),) or not np.isfinite(value).all():
        raise ValueError('finite probability-scale effect predictions required')
    return value


def recovery(scores, mask):
    mask, scores = np.asarray(mask,dtype=bool),np.asarray(scores,dtype=np.float64)
    if mask.shape!=scores.shape:
        raise ValueError('modifier mask and raw feature scores disagree')
    if not mask.any():
        return dict(applicable=False,auroc=None,expected_topk_hits=None,
                    expected_topk_precision=None,probability_all_signals_selected=None,
                    reason='null scenario has no true modifiers; ranking is not a selection test')
    # Existing tie-aware routine uses a leading true-signal block; reorder solely
    # inside evaluator metrics, never in model fitting or summary construction.
    order=np.r_[np.flatnonzero(mask),np.flatnonzero(~mask)]
    return dict(applicable=True,**signal_recovery_with_ties(scores[order],int(mask.sum()),'probability_tau'))


def exact_scores(model,x,plan,squared=False):
    baseline=predictions(model,x)
    scores=np.zeros(x.shape[1],dtype=np.float64)
    for j,_,changed in perturbed_covariates(x,plan):
        delta=baseline-predictions(model,changed)
        scores[j]+=float(np.mean(delta*delta)/2 if squared else np.mean(np.abs(delta)))/plan.shape[1]
    return scores


def external_test(model,evaluation,seed,config,legacy_signal_dim=2):
    x=evaluation.rows.x
    tau=predictions(model,x)
    if evaluation.site_ids is None:
        plan=make_permutations(len(x),x.shape[1],config['test_seed_offset']+seed,config['repeats'])
        scores=exact_scores(model,x,plan)
        hashes=[permutation_hash(plan)]
        observation_weights=np.full(len(x),1/len(x))
        perturbation_scope='legacy_pooled_common_external_test'
    else:
        observation_weights=np.zeros(len(x))
        scores=np.zeros(x.shape[1])
        hashes=[]
        for site_id,weight in enumerate(evaluation.target_weights):
            indices=np.flatnonzero(evaluation.site_ids==site_id)
            if len(indices)<2:
                raise ValueError('external target-mixture site needs at least two sampled rows')
            plan=make_permutations(len(indices),x.shape[1],
                config['test_seed_offset']+seed+1000003*site_id,config['repeats'])
            scores+=weight*exact_scores(model,x[indices],plan)
            observation_weights[indices]=weight/len(indices)
            hashes.append(permutation_hash(plan))
        perturbation_scope='within_external_test_site_target_pi'
    truth=evaluation.truth
    mask=truth.modifier_mask if truth is not None else np.arange(x.shape[1])<legacy_signal_dim
    rmse=float(np.sqrt(observation_weights@((tau-truth.tau)**2))) if truth is not None else None
    mean_tau=float(observation_weights@tau)
    return dict(score_definition='external_mean_absolute_tau_change',scores=scores.tolist(),
                recovery=recovery(scores,mask),cate_rmse=rmse,sqrt_pehe=rmse,
                pehe_convention='PEHE=mean((tau_hat-tau_true)^2); sqrt_PEHE=CATE_RMSE',
                true_cate_available=truth is not None,estimated_heterogeneity=float(observation_weights@((tau-mean_tau)**2)),
                effect_out_of_bounds_fraction=float(observation_weights@(np.abs(tau)>1)),
                prediction_clipped=False,test_rows=len(x),raw_x_sha256=array_hash(x),
                perturbation_scope=perturbation_scope,permutation_sha256=hashes)


def holdout_reference(model,sites,plans,modifier_mask):
    if not sites or len(sites)!=len(plans):
        raise ValueError('one holdout perturbation plan per site required')
    counts=np.asarray([s.n_total for s in sites],dtype=np.float64)
    weights=counts/counts.sum()
    local=[exact_scores(model,s.holdout.x,p,squared=True) for s,p in zip(sites,plans)]
    scores=sum(w*v for w,v in zip(weights,local))
    return dict(score_definition='target_weighted_H_half_mean_squared_tau_change',
                exact_squared_scores=scores.tolist(),recovery=recovery(scores,modifier_mask),
                target_weights=weights.tolist(),local_mean_scores=[v.tolist() for v in local],
                rows=[len(s.holdout) for s in sites],
                permutation_sha256=[permutation_hash(p) for p in plans])


def _top2_distribution(values):
    values=np.asarray(values,dtype=float)
    k=min(2,len(values))
    cutoff=np.sort(values)[-k]
    above=tuple(np.flatnonzero(values>cutoff))
    tied=np.flatnonzero(values==cutoff)
    slots=k-len(above)
    probability=1/math.comb(len(tied),slots)
    return {tuple(sorted((*above,*choice))):probability for choice in combinations(tied,slots)}


def _average_ranks(values):
    values=np.asarray(values)
    result=np.empty(len(values),dtype=float)
    for v in np.unique(values):
        result[values==v]=np.sum(values<v)+(np.sum(values==v)+1)/2
    return result


def score_agreement(exact,approximate):
    exact,approximate=np.asarray(exact,dtype=float),np.asarray(approximate,dtype=float)
    if exact.shape!=approximate.shape or exact.ndim!=1 or not len(exact):
        raise ValueError('aligned score vectors required')
    if not np.isfinite(exact).all() or not np.isfinite(approximate).all():
        raise ValueError('finite score vectors required')
    errors=approximate-exact
    a,b=_average_ranks(exact),_average_ranks(approximate)
    correlation=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    first,second=_top2_distribution(exact),_top2_distribution(approximate)
    same=sum(probability*second.get(selection,0) for selection,probability in first.items())
    return dict(per_variable_error=errors.tolist(),maximum_absolute_error=float(np.max(np.abs(errors))),
                mean_absolute_error=float(np.mean(np.abs(errors))),spearman_rank_correlation=correlation,
                top2_set_agreement_probability=same,
                top2_tie_rule='independent uniform selection within exact boundary ties',
                correlation_undefined_reason='constant ranking' if correlation is None else None)


def linearity_gate(exact_tau,linear_tau,exact_scores_value,linear_scores,gate):
    exact_tau,linear_tau=np.asarray(exact_tau),np.asarray(linear_tau)
    floor=gate['denominator_floor']
    relative_tau=float(np.sqrt(np.mean((linear_tau-exact_tau)**2))/max(
        floor,float(np.sqrt(np.mean(exact_tau**2)))))
    relative_score=float(np.max(np.abs(np.asarray(linear_scores)-exact_scores_value))/max(
        floor,float(np.max(np.abs(exact_scores_value)))))
    return dict(compression_eligible=relative_tau<=gate['relative_tau_rmse_max']
                and relative_score<=gate['relative_score_linf_max'],
                relative_tau_rmse=relative_tau,relative_score_linf=relative_score,
                thresholds=dict(gate), interpretation='engineering expansion gate; keep failed linearization diagnostics')


def summarize_linearization(local_diagnostics,weights,mask,gate):
    """Combine site means once; full-gradient evidence precedes compression."""
    weights=np.asarray(weights,dtype=float)
    if (not local_diagnostics or weights.shape!=(len(local_diagnostics),)
            or np.any(weights<=0) or not np.isclose(weights.sum(),1,rtol=0,atol=1e-14)):
        raise ValueError('aligned positive target weights required')
    names=set(local_diagnostics[0]['scores'])
    if any(set(item['scores'])!=names for item in local_diagnostics):
        raise ValueError('site diagnostic scopes differ')
    scores={name:weights@np.asarray([item['scores'][name] for item in local_diagnostics]) for name in names}
    exact_rms=float(np.sqrt(weights@np.asarray([
        item['exact_tau_mean_square_over_original_and_all_interventions'] for item in local_diagnostics])))
    errors={name:float(np.sqrt(weights@np.square([
        item['tau_rmse_over_original_and_all_interventions'][name] for item in local_diagnostics])))
        for name in names-{'exact'}}
    floor=gate['denominator_floor']
    relative_tau=errors['full_gradient']/max(floor,exact_rms)
    relative_score=float(np.max(np.abs(scores['full_gradient']-scores['exact'])))/max(
        floor,float(np.max(np.abs(scores['exact']))))
    return dict(scores={k:v.tolist() for k,v in scores.items()},
        tau_rmse=errors,exact_tau_rms=exact_rms,
        score_agreement={k:score_agreement(scores['exact'],v) for k,v in scores.items() if k!='exact'},
        recovery={k:recovery(v,mask) for k,v in scores.items()},
        gate=dict(compression_eligible=relative_tau<=gate['relative_tau_rmse_max']
                  and relative_score<=gate['relative_score_linf_max'],
                  relative_tau_rmse=relative_tau,relative_score_linf=relative_score,
                  thresholds=dict(gate),interpretation='engineering expansion gate; keep failed linearization diagnostics'))
