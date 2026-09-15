"""The five fixed seed-zero replays; no training or expansion recipes."""
import json
from pathlib import Path

PILOT_TASK = 'effect-modifier-effects-replay-pilot'
PROTOCOL = 'effects-replay-v1'
BINDINGS_PATH = Path(__file__).with_name('source_bindings.json')


def cohort_plan():
    return [dict(recipe='fisher_'+model, dataset=dataset)
            for dataset in ('scmv2_base', 'legacy_control_iid')
            for model in ('shallow', 'deep')] + [dict(recipe='dr_path', dataset='scmv2_base')]


def validate_parameters(task, parameters):
    if task != PILOT_TASK or not isinstance(parameters, dict) or set(parameters) != {'recipe', 'dataset'}:
        raise ValueError('effects replay requires exactly recipe/dataset')
    if parameters not in cohort_plan():
        raise ValueError('not one of the five reviewed replay cells')
    return dict(parameters)


def run_id(task, parameters):
    p = validate_parameters(task, parameters)
    return 'ose_replay_v1_'+p['recipe']+'_'+p['dataset']+'_pilot'


def source_bindings():
    if BINDINGS_PATH.is_symlink() or BINDINGS_PATH.resolve() != BINDINGS_PATH:
        raise ValueError('source bindings cannot be a symlink')
    return json.loads(BINDINGS_PATH.read_text())


def run_config(task, parameters):
    p = validate_parameters(task, parameters)
    binding = source_bindings()
    recipes = ({'linear': 'dr_linear', 'spline': 'dr_spline'} if p['recipe'] == 'dr_path'
               else {key: key+'_'+p['recipe'].split('_', 1)[1] for key in ('aggregation', 'linearity')})
    inputs = {role: binding['sources']['ose_v1_'+recipe+'_'+p['dataset']+'_pilot']
              for role, recipe in recipes.items()}
    return dict(protocol=PROTOCOL, task=task, **p, run_id=run_id(task, p), seed=0,
        sources=inputs, accepted_report=binding['accepted_report'],
        original_source_provenance=binding['source_provenance'],
        original_generation_source_files=binding['original_generation_source_files'],
        t_grid=[0., .001, .01, .03, .1, .25, .5, .75, 1.],
        fd_steps=[1e-4, 1e-5, 1e-6], sample_rows_per_site=32,
        sample_seed_stream=8600021, direction_seed_stream=8700023,
        seed_derivation='SeedSequence([seed, stream, site_id]); independent of predictions and architecture',
        fd_atol=1e-7, fd_rtol=1e-4,
        replay_tolerance=dict(rtol=1e-9, atol=1e-10),
        gradient_gate='stable activation entries only; abs(error)<=atol+rtol*abs(autograd); report boundary and unavailable cases',
        data_gate='exact original T/V/H, test and five-repeat permutation fingerprints before inference',
        selection_rule='retain original validation-selected lambda and saved checkpoints; test curves are diagnostic only',
        expansion_authorized=False, training_tasks=0, fitting_allowed=False,
        communication_scope='evaluator replay of saved artifacts; no new deployable one-upload method',
        future_gate=dict(implementation_required=True, full_holdout_relative_tau_max=.25,
                         full_holdout_relative_score_max=.25,
                         cate_quality='must not clearly degrade; review paired diagnostics before freezing a rule',
                         fresh_seed_confirmation='only after training and selection rules are frozen'))
