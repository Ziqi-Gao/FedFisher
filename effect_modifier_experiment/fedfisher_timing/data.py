"""Preserve the original generator's global row order for validation sampling."""
from data import get_dataset


def generate_datasets(config, seed, split):
    d = config['dgp']
    return get_dataset(
        datatype='SyntheticEffectModifier', n_client=d['num_sites'], n_c=2,
        alpha=d['alpha'], synthetic_split=split, synthetic_num_test=d['num_test'],
        effect_modifier_covariate_dim=d['covariate_dim'],
        effect_modifier_signal_dim=d['signal_dim'],
        effect_modifier_signal_strength=d['signal_strength'],
        effect_modifier_main_effect_strength=d['main_effect_strength'],
        effect_modifier_confounding_strength=d['confounding_strength'],
        effect_modifier_intercept=d['outcome_intercept'],
        effect_modifier_treatment_prob=d['treatment_prob'],
        effect_modifier_propensity_clip=d['propensity_clip'],
        effect_modifier_site_min_samples=d['site_min_samples'],
        effect_modifier_site_max_samples=d['site_max_samples'],
        effect_modifier_site_gaussian_shift='site_gaussian' in config['variant'],
        effect_modifier_site_gaussian_mean_span=d['site_gaussian_mean_span'],
        effect_modifier_site_gaussian_std_low=d['site_gaussian_std_low'],
        effect_modifier_site_gaussian_std_high=d['site_gaussian_std_high'], seed=seed)
