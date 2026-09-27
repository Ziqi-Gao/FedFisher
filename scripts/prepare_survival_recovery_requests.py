#!/usr/bin/env python3
"""Create immutable recovery outbox requests; never submit central jobs."""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
sys.path.insert(0, str(PROJECT_ROOT))

from effect_modifier_experiment.infrastructure.provenance import compute_provenance
from effect_modifier_experiment.infrastructure.requests import (
    canonical, encoded, sha256, clean_provenance, review_sources as source_bindings,
    atomic_create, publish_bundle)
from effect_modifier_experiment.survival import recovery_protocol as recovery

OUTBOX_ROOT = Path('/scr/del6500/FedFisher/scheduler/outbox')
REPORT_ROOT = Path('/scr/del6500/FedFisher/scheduler/reports')
REVIEW_SOURCES = {name: PROJECT_ROOT / path for name, path in {
    'proposal': 'effect_modifier_experiment/server_scheduler/survival-recovery-v1/registration-proposal.toml',
    'diagnostic_schema': 'effect_modifier_experiment/server_scheduler/survival-recovery-v1/diagnostic-parameters.schema.json',
    'formal_schema': 'effect_modifier_experiment/server_scheduler/survival-recovery-v1/shard-parameters.schema.json',
    'adapter': 'scripts/fedfisher_server_scheduler_adapter.py',
    'recovery': 'effect_modifier_experiment/survival/recovery.py',
    'protocol': 'effect_modifier_experiment/survival/recovery_protocol.py',
    'method': 'effect_modifier_experiment/survival/recovery_method.py',
    'dslasso': 'effect_modifier_experiment/survival/dslasso.py',
    'artifacts': 'effect_modifier_experiment/survival/artifacts.py',
    'acceptance': 'effect_modifier_experiment/survival/recovery_acceptance.py',
    'request_builder': 'scripts/prepare_survival_recovery_requests.py',
    'acceptance_cli': 'scripts/validate_survival_recovery.py',
    'description': 'effect_modifier_experiment/survival/README.md',
}.items()}


def bundle_prefix(epoch=1):
    recovery.tasks_for_epoch(epoch)
    return 'fedfisher-survival-dslasso-' + {
        1: 'recovery-v1', 2: 'repair-v2', 3: 'integrated-v3'}[epoch]


def bundle_name(generation=1, cohort='diagnostic', epoch=1):
    if type(generation) is not int or generation < 1 or cohort not in ('diagnostic', 'formal'):
        raise ValueError('invalid recovery generation or cohort')
    return '%s-%s-p%d' % (bundle_prefix(epoch), cohort, generation)


def build_requests(generation=1, cohort='diagnostic', shard_count=None, *, recipe=None, epoch=1):
    """Pure specs only. Formal publication independently replays actual acceptance."""
    prefix = bundle_name(generation, cohort, epoch)
    diagnostic_task, formal_task = recovery.tasks_for_epoch(epoch)
    cases = recovery.cases_for_task(diagnostic_task)
    diagnostic = cohort == 'diagnostic'
    if diagnostic:
        if shard_count is not None or recipe is not None:
            raise ValueError('diagnostic has no shard or selected-recipe override')
        task = diagnostic_task
        parameters = ([dict(case=case, recipe=candidate)
                       for case in cases for candidate in recovery.RECIPES] if epoch == 1
                      else [dict(case=case) for case in cases])
    else:
        count = 16 if shard_count is None else shard_count
        if type(count) is not int or count not in (8, 16) or recipe not in recovery.RECIPES[1:]:
            raise ValueError('formal specs require a candidate recipe and 8/16 deterministic shards')
        if epoch >= 2 and recipe != recovery.REPAIR_RECIPE:
            raise ValueError('repair formal specs require the fixed previously selected recipe')
        task = formal_task
        parameters = [dict(scenario=scenario, shard_index=index, shard_count=count)
                      for scenario in ('iid', 'covariate_shift') for index in range(count)]
    specs, coverage = [], {}
    for p in parameters:
        recovery.validate_parameters(task, p)
        suffix = ((('%s-%s' % (p['case'], p['recipe'])) if epoch == 1 else p['case']) if diagnostic else
                  '%s-s%02dof%02d' % (p['scenario'], p['shard_index'], p['shard_count']))
        job_id = prefix + '-' + suffix.replace('_', '-')
        if re.fullmatch(r'fedfisher-survival-[A-Za-z0-9_.-]{1,109}', job_id) is None:
            raise ValueError('invalid recovery job identifier')
        request = dict(schema_version=2, project='FedFisher', task=task,
                       job_id=job_id, parameters=p, priority=0)
        config = json.loads(canonical(recovery.run_config(task, p, recipe=recipe)))
        seeds = list(recovery.seed_sequence(task, p))
        key = (p['case'], p.get('recipe', recovery.REPAIR_RECIPE)) if diagnostic else p['scenario']
        coverage.setdefault(key, []).extend(seeds)
        specs.append(dict(filename=job_id + '.job-request.json', request=request,
            request_sha256=sha256(encoded(request)), run_id=config['run_id'], configuration=config,
            configuration_sha256=sha256(canonical(config).encode()), seed_sequence=seeds))
    if diagnostic:
        if len(coverage) != len(parameters) or any(seeds != [cases[key[0]]['seed']]
                                     for key, seeds in coverage.items()):
            raise ValueError('diagnostic must cover every fixed case and recipe exactly once')
    elif len(coverage) != 2 or any(sorted(seeds) != list(range(300)) for seeds in coverage.values()):
        raise ValueError('formal recovery must cover exactly 600 cells')
    return tuple(specs)


def current_provenance():
    return clean_provenance(compute_provenance(PROJECT_ROOT))


def review_sources(epoch=1):
    recovery.tasks_for_epoch(epoch)
    sources = dict(REVIEW_SOURCES)
    if epoch >= 2:
        for key in ('proposal', 'diagnostic_schema', 'formal_schema'):
            directory = 'survival-repair-v2' if epoch == 2 else 'survival-integrated-v3'
            sources[key] = sources[key].parent.parent / directory / sources[key].name
    return source_bindings(sources, PROJECT_ROOT)


def acceptance_binding(cohort, path, provenance, epoch=1):
    from effect_modifier_experiment.survival import recovery_acceptance as gate
    gate.check_original_closeout()
    if epoch >= 2:
        gate.check_repair_closeout()
    if epoch == 3:
        gate.check_integration_reference()
    if cohort == 'diagnostic':
        if path is not None:
            raise ValueError('diagnostic publication must not take a prerequisite override')
        return None
    if path is None or Path(path) != gate.default_acceptance_path(provenance, epoch=epoch):
        raise ValueError('formal publication requires actual source-bound diagnostic acceptance first')
    proof = gate.check_current_diagnostic_gate(epoch=epoch)
    if proof['provenance'] != provenance:
        raise ValueError('diagnostic source differs from current clean source')
    return dict(path=str(path), sha256=sha256(Path(path).read_bytes()), evidence=proof)


def write_requests(generation=1, cohort='diagnostic', diagnostic_acceptance=None, shard_count=None, epoch=1):
    name = bundle_name(generation, cohort, epoch)
    before = current_provenance()
    accepted = acceptance_binding(cohort, diagnostic_acceptance, before, epoch)
    recipe = None if accepted is None else accepted['evidence']['selected_recipe']
    specs = build_requests(generation, cohort, shard_count, recipe=recipe, epoch=epoch)
    metadata = dict(schema_version=1, bundle=bundle_prefix(epoch) + '-' + cohort,
        generation=generation, task=specs[0]['request']['task'], scientific_cell_count=len(specs) if cohort == 'diagnostic' else 600,
        shard_count=None if cohort == 'diagnostic' else (16 if shard_count is None else shard_count),
        original_closeout=recovery.ORIGINAL_CLOSEOUT, case_inventory=recovery.CASE_INVENTORY,
        selected_recipe=recipe, formal_publication_authorized=cohort == 'formal',
        operator_gate='review disabled proposal, fixtures and profile; centrally admit diagnostic first; '
                      'formal requires actual same-source diagnostic acceptance and operator cost review',
        freeze_rule='keep canonical source unchanged through diagnostic, formal and full scientific acceptance')
    if epoch >= 2:
        metadata.update(epoch=epoch, repair_closeout=recovery.REPAIR_CLOSEOUT,
                        r1_diagnostic_acceptance=recovery.R1_DIAGNOSTIC_ACCEPTANCE)
    if epoch == 3:
        metadata['r2_diagnostic_acceptance'] = recovery.R2_DIAGNOSTIC_ACCEPTANCE
    return publish_bundle(specs, bundle=OUTBOX_ROOT / name,
        inventory_path=REPORT_ROOT / (name + '.inventory.json'), metadata=metadata,
        provenance=before, sources=review_sources(epoch), acceptance=accepted,
        check_provenance=current_provenance, check_sources=lambda: review_sources(epoch),
        check_acceptance=lambda: acceptance_binding(cohort, diagnostic_acceptance, before, epoch), create=atomic_create)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--epoch', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--cohort', choices=('diagnostic', 'formal'), default='diagnostic')
    parser.add_argument('--diagnostic-acceptance', type=Path)
    parser.add_argument('--shard-count', type=int, choices=(8, 16))
    args = parser.parse_args(argv)
    result = write_requests(args.generation, args.cohort, args.diagnostic_acceptance, args.shard_count, args.epoch)
    print(json.dumps(dict(inventory=result['inventory_path'], request_count=result['request_count'],
                         scientific_cell_count=result['scientific_cell_count'], submitted=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
