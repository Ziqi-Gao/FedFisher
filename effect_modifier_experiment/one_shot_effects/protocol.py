"""Fixed staged pilot recipes; requests express science and select no hardware."""
from dataclasses import asdict, replace

PILOT_TASK = 'effect-modifier-one-shot-effects-pilot'
PROTOCOL = 'one-shot-effects-v1'
SCENARIOS = ('base','covariate_shift','correlated_shift','interaction','null')
LEGACY_VARIANTS = ('control','site_gaussian_shift','local_permutation','site_gaussian_local_permutation')
DATASETS = { 'scmv2_'+s:dict(family='scmv2',scenario=s) for s in SCENARIOS }
DATASETS.update({'legacy_'+v+'_'+s:dict(family='legacy',variant=v,split=s)
                 for v in LEGACY_VARIANTS for s in ('iid','noniid')})
RECIPES = {
    'dr_spline':dict(family='dr_sieve',basis='additive_cubic_bspline',stage=2),
    'dr_linear':dict(family='dr_sieve',basis='linear',stage=2),
    'r_unpenalized':dict(family='r_common',penalty='none',stage=2),
    'r_adaptive_lasso':dict(family='r_common',penalty='adaptive_lasso',stage=2),
}
for short, model in (('shallow','SyntheticMLP'),('deep','SyntheticMLPDeep')):
    for family in ('aggregation','linearity','fedavg'):
        RECIPES[family+'_'+short] = dict(family=family,model=model,stage=2)
    for compression in ('fisher_gradient','euclidean_gradient','random_orthogonal'):
        for rank in (16,32,64):
            RECIPES['compression_'+short+'_'+compression+'_r'+str(rank)] = dict(
                family='compression',model=model,compression=compression,rank=rank,stage=3)


def validate_parameters(task, parameters):
    if task != PILOT_TASK or not isinstance(parameters,dict) or set(parameters) != {'recipe','dataset'}:
        raise ValueError('one-shot effects pilot requires exactly recipe/dataset')
    if parameters['recipe'] not in RECIPES or parameters['dataset'] not in DATASETS:
        raise ValueError('unknown staged pilot recipe or dataset')
    return dict(parameters)


def run_id(task, parameters):
    p = validate_parameters(task,parameters)
    return 'ose_v1_'+p['recipe']+'_'+p['dataset']+'_pilot'


def run_config(task, parameters):
    from .data import SCMConfig, LegacyConfig, PARTITION_RULE, PARTITION_SEED_STREAM
    from .dr_nuisance import DRNuisanceConfig
    from .dr_sieve import DRSieveConfig
    from .fedfisher_em import FedFisherEMConfig, LegacyDiagnosticConfig
    from ..causal_comparison.rlearner import RlearnerConfig
    p = validate_parameters(task,parameters)
    recipe, dataset = dict(RECIPES[p['recipe']]), dict(DATASETS[p['dataset']])
    fisher = FedFisherEMConfig()
    if recipe['family']=='compression':
        fisher = replace(fisher,rank=recipe['rank'],compression=recipe['compression'])
    return dict(protocol=PROTOCOL,task=task,**p,run_id=run_id(task,p),seed=0,
        recipe_definition=recipe,dataset_definition=dataset,
        dgp=asdict(SCMConfig() if dataset['family']=='scmv2' else LegacyConfig()),
        split=dict(rule=PARTITION_RULE,seed_stream=PARTITION_SEED_STREAM),
        target_weights='original_site_n_total/sum_original_site_n_total; each split is a site mean',
        fisher=asdict(fisher),legacy_diagnostic=asdict(LegacyDiagnosticConfig()),
        dr_nuisance=asdict(DRNuisanceConfig()),dr_sieve=asdict(DRSieveConfig()),
        r_baseline=asdict(RlearnerConfig()),
        upload_byte_budget_rule='compression: round_up_1024(8*(D*(rank+2)+p*(rank+1)^2)+65536); charge padding; native bytes also reported',
        evaluation=dict(repeats=5,holdout_seed_offset=8400011,test_seed_offset=8500009,
                        external_perturbation_by_dataset='SCMV2_within_test_site_target_pi; legacy_pooled_common_test',
                        external_score='mean_absolute_tau_change',
                        summary_score='half_mean_squared_tau_change',
                        perturbation='within_site_marginal_permutation'),
        linearity_gate=dict(relative_tau_rmse_max=0.25,relative_score_linf_max=0.25,
                            denominator_floor=1e-8,
                            role='predeclared engineering pilot expansion gate; no statistical guarantee'),
        execution_scope='bounded_seed_0_only; no formal task registered by this proposal')


def linearity_prerequisite(config):
    if config['recipe_definition']['family']!='compression':
        return None
    short = 'shallow' if config['recipe_definition']['model']=='SyntheticMLP' else 'deep'
    return run_config(PILOT_TASK,dict(recipe='linearity_'+short,dataset=config['dataset']))


def staged_pilot_plan():
    """Reviewable small rollout, deliberately not the full Cartesian product."""
    return dict(
        stage2=[dict(recipe=r,dataset=d) for d in ('scmv2_base','legacy_control_iid')
                for r in ('aggregation_shallow','aggregation_deep','linearity_shallow',
                          'linearity_deep','dr_linear','dr_spline')],
        stage3_after_matching_linearity_gate=[dict(recipe=r,dataset='scmv2_base')
                                             for r,v in RECIPES.items() if v['stage']==3],
        later_small_scenario_checks=[dict(recipe=r,dataset='scmv2_'+s)
                                    for s in SCENARIOS for r in ('dr_linear','dr_spline','r_unpenalized',
                                                                 'r_adaptive_lasso','fedavg_shallow')],
        formal='Not prepared: requires accepted pilots, reviewed source and a separate formal contract.')


def compression_wire_budget(config):
    """Public-shape budget shared across three direction strategies at each rank.

    Actual padding bytes are transmitted and counted. This compares the same
    rank at exactly the same byte budget; it is not optimal variable-rank coding.
    """
    from models import SYNTHETIC_MLP_HIDDEN_DIMS
    r=config['recipe_definition']
    if r['family']!='compression':
        return None
    d,rank=config['dgp']['covariate_dim'],r['rank']
    widths=(1+2*d,*SYNTHETIC_MLP_HIDDEN_DIMS[r['model']],2)
    parameters=sum(a*b+b for a,b in zip(widths[:-1],widths[1:]))
    raw_bound=8*(parameters*(rank+2)+d*(rank+1)**2)
    return ((raw_bound+65536+1023)//1024)*1024
