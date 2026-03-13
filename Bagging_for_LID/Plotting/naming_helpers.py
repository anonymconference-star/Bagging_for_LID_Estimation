import pandas as pd
###################################################OWN IMPORT###########################################################
from Bagging_for_LID.Plotting.plotting_helpers import *
from Bagging_for_LID.Plotting.optimize_across_parameter_results import *

#!!Hard coded part!! This is for changing the full, class parameter based identifiers into the expressive naming conventiones used in the paper.
def unordered_lookup(query, original_map = None, sep= '|'):
    if original_map is None:
        original_map  =  {
    'bagging_method:bag | pre_smooth:False | post_smooth:False': 'Bagging',
    'bagging_method:bag | pre_smooth:False | post_smooth:True': 'Bagging with post-smoothing',
    'bagging_method:bag | pre_smooth:True | post_smooth:False': 'Bagging with pre-smoothing',
    'bagging_method:bag | pre_smooth:True | post_smooth:True': 'Bagging with pre-smoothing and post-smoothing',
    'bagging_method:None | pre_smooth:False | post_smooth:False': 'Baseline',
    'bagging_method:None | pre_smooth:False | post_smooth:True': 'Baseline with smoothing'}
    def build_canonical_map(original: dict[str, str], sep: str = '|') -> dict[tuple[str, ...], str]:
        return {
            tuple(sorted(part.strip() for part in key.split(sep))): value
            for key, value in original.items()
        }
    canonical_map = build_canonical_map(original_map)
    signature = tuple(sorted(part.strip() for part in query.split(sep)))
    return canonical_map.get(signature)

#!!Hard coded part!! This is for changing the full, class parameter based identifiers into the expressive naming conventiones used in the paper.
def modify_label(label):
    if label == 'bagging_method:bag':
        label = 'Bagging'
    elif label == 'bagging_method:bagw':
        label = 'Bagging with out-of-bag weights'
    elif label == 'bagging_method:bagwth':
        label = 'Bagging with out-of-bag weights (adjust)'
    elif label == 'bagging_method:approx_bagwth':
        label = 'Bagging with out-of-bag weights (approximate adjust)'
    elif label == 'bagging_method:None':
        label = 'Baseline'
    else:
        label = unordered_lookup(label)
    return label

#!!Hard coded part!! This if for reordering experiments, so that different plots get the same colored spider chart lines (based on method variants), and they are arranged in a logical order in the table, instead of random.
def reorder_sorted_experiments(df, order=None, keep_rest=True, sweep_params=None):
    order_mle = [(('Nbag', 10),
                  ('bagging_method', None),
                  ('estimator_name', 'mle'),
                  ('post_smooth', False),
                  ('pre_smooth', False),
                  ('submethod_0', '0'),
                  ('submethod_error', 'log_diff'),
                  ('t', 1)), (('Nbag', 10),
                              ('bagging_method', None),
                              ('estimator_name', 'mle'),
                              ('post_smooth', True),
                              ('pre_smooth', False),
                              ('submethod_0', '0'),
                              ('submethod_error', 'log_diff'),
                              ('t', 1)), (('Nbag', 10),
                                          ('bagging_method', 'bag'),
                                          ('estimator_name', 'mle'),
                                          ('post_smooth', False),
                                          ('pre_smooth', False),
                                          ('submethod_0', '0'),
                                          ('submethod_error', 'log_diff'),
                                          ('t', 1)), (('Nbag', 10),
                                                      ('bagging_method', 'bag'),
                                                      ('estimator_name', 'mle'),
                                                      ('post_smooth', True),
                                                      ('pre_smooth', False),
                                                      ('submethod_0', '0'),
                                                      ('submethod_error', 'log_diff'),
                                                      ('t', 1)), (('Nbag', 10),
                                                                  ('bagging_method', 'bag'),
                                                                  ('estimator_name', 'mle'),
                                                                  ('post_smooth', False),
                                                                  ('pre_smooth', True),
                                                                  ('submethod_0', '0'),
                                                                  ('submethod_error', 'log_diff'),
                                                                  ('t', 1)), (('Nbag', 10),
                                                                              ('bagging_method', 'bag'),
                                                                              ('estimator_name', 'mle'),
                                                                              ('post_smooth', True),
                                                                              ('pre_smooth', True),
                                                                              ('submethod_0', '0'),
                                                                              ('submethod_error', 'log_diff'),
                                                                              ('t', 1)), (('Nbag', 10),
                                                                                          ('bagging_method', 'bagw'),
                                                                                          ('estimator_name', 'mle'),
                                                                                          ('post_smooth', False),
                                                                                          ('pre_smooth', False),
                                                                                          ('submethod_0', '0'),
                                                                                          ('submethod_error',
                                                                                           'log_diff'),
                                                                                          ('t', 1)), (('Nbag', 10),
                                                                                                      ('bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                      ('estimator_name',
                                                                                                       'mle'),
                                                                                                      ('post_smooth',
                                                                                                       False),
                                                                                                      ('pre_smooth',
                                                                                                       False),
                                                                                                      ('submethod_0',
                                                                                                       '0'),
                                                                                                      (
                                                                                                      'submethod_error',
                                                                                                      'log_diff'),
                                                                                                      ('t', 1))]

    order_tle = [(('Nbag', 10),
                  ('bagging_method', None),
                  ('estimator_name', 'tle'),
                  ('post_smooth', False),
                  ('pre_smooth', False),
                  ('submethod_0', '0'),
                  ('submethod_error', 'log_diff'),
                  ('t', 1)), (('Nbag', 10),
                              ('bagging_method', None),
                              ('estimator_name', 'tle'),
                              ('post_smooth', True),
                              ('pre_smooth', False),
                              ('submethod_0', '0'),
                              ('submethod_error', 'log_diff'),
                              ('t', 1)), (('Nbag', 10),
                                          ('bagging_method', 'bag'),
                                          ('estimator_name', 'tle'),
                                          ('post_smooth', False),
                                          ('pre_smooth', False),
                                          ('submethod_0', '0'),
                                          ('submethod_error', 'log_diff'),
                                          ('t', 1)), (('Nbag', 10),
                                                      ('bagging_method', 'bag'),
                                                      ('estimator_name', 'tle'),
                                                      ('post_smooth', True),
                                                      ('pre_smooth', False),
                                                      ('submethod_0', '0'),
                                                      ('submethod_error', 'log_diff'),
                                                      ('t', 1)), (('Nbag', 10),
                                                                  ('bagging_method', 'bag'),
                                                                  ('estimator_name', 'tle'),
                                                                  ('post_smooth', False),
                                                                  ('pre_smooth', True),
                                                                  ('submethod_0', '0'),
                                                                  ('submethod_error', 'log_diff'),
                                                                  ('t', 1)), (('Nbag', 10),
                                                                              ('bagging_method', 'bag'),
                                                                              ('estimator_name', 'tle'),
                                                                              ('post_smooth', True),
                                                                              ('pre_smooth', True),
                                                                              ('submethod_0', '0'),
                                                                              ('submethod_error', 'log_diff'),
                                                                              ('t', 1)), (('Nbag', 10),
                                                                                          ('bagging_method', 'bagw'),
                                                                                          ('estimator_name', 'tle'),
                                                                                          ('post_smooth', False),
                                                                                          ('pre_smooth', False),
                                                                                          ('submethod_0', '0'),
                                                                                          ('submethod_error',
                                                                                           'log_diff'),
                                                                                          ('t', 1)), (('Nbag', 10),
                                                                                                      ('bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                      ('estimator_name',
                                                                                                       'tle'),
                                                                                                      ('post_smooth',
                                                                                                       False),
                                                                                                      ('pre_smooth',
                                                                                                       False),
                                                                                                      ('submethod_0',
                                                                                                       '0'),
                                                                                                      (
                                                                                                      'submethod_error',
                                                                                                      'log_diff'),
                                                                                                      ('t', 1))]

    order_mada = [(('Nbag', 10),
                   ('bagging_method', None),
                   ('estimator_name', 'mada'),
                   ('post_smooth', False),
                   ('pre_smooth', False),
                   ('submethod_0', '0'),
                   ('submethod_error', 'log_diff'),
                   ('t', 1)), (('Nbag', 10),
                               ('bagging_method', None),
                               ('estimator_name', 'mada'),
                               ('post_smooth', True),
                               ('pre_smooth', False),
                               ('submethod_0', '0'),
                               ('submethod_error', 'log_diff'),
                               ('t', 1)), (('Nbag', 10),
                                           ('bagging_method', 'bag'),
                                           ('estimator_name', 'mada'),
                                           ('post_smooth', False),
                                           ('pre_smooth', False),
                                           ('submethod_0', '0'),
                                           ('submethod_error', 'log_diff'),
                                           ('t', 1)), (('Nbag', 10),
                                                       ('bagging_method', 'bag'),
                                                       ('estimator_name', 'mada'),
                                                       ('post_smooth', True),
                                                       ('pre_smooth', False),
                                                       ('submethod_0', '0'),
                                                       ('submethod_error', 'log_diff'),
                                                       ('t', 1)), (('Nbag', 10),
                                                                   ('bagging_method', 'bag'),
                                                                   ('estimator_name', 'mada'),
                                                                   ('post_smooth', False),
                                                                   ('pre_smooth', True),
                                                                   ('submethod_0', '0'),
                                                                   ('submethod_error', 'log_diff'),
                                                                   ('t', 1)), (('Nbag', 10),
                                                                               ('bagging_method', 'bag'),
                                                                               ('estimator_name', 'mada'),
                                                                               ('post_smooth', True),
                                                                               ('pre_smooth', True),
                                                                               ('submethod_0', '0'),
                                                                               ('submethod_error', 'log_diff'),
                                                                               ('t', 1)), (('Nbag', 10),
                                                                                           ('bagging_method', 'bagw'),
                                                                                           ('estimator_name', 'mada'),
                                                                                           ('post_smooth', False),
                                                                                           ('pre_smooth', False),
                                                                                           ('submethod_0', '0'),
                                                                                           ('submethod_error',
                                                                                            'log_diff'),
                                                                                           ('t', 1)), (('Nbag', 10),
                                                                                                       (
                                                                                                       'bagging_method',
                                                                                                       'approx_bagwth'),
                                                                                                       (
                                                                                                       'estimator_name',
                                                                                                       'mada'),
                                                                                                       ('post_smooth',
                                                                                                        False),
                                                                                                       ('pre_smooth',
                                                                                                        False),
                                                                                                       ('submethod_0',
                                                                                                        '0'),
                                                                                                       (
                                                                                                       'submethod_error',
                                                                                                       'log_diff'),
                                                                                                       ('t', 1))]
    order_mle[1], order_mle[2] = order_mle[2], order_mle[1]
    order_mada[1], order_mada[2] = order_mada[2], order_mada[1]
    order_tle[1], order_tle[2] = order_tle[2], order_tle[1]
    default_order = order_mle + order_tle + order_mada
    order = default_order if order is None else order
    sweep_set = set(sweep_params or [])
    def _reduced(tpl):
        return tuple(pair for pair in tpl if pair[0] not in sweep_set)
    cols_list = list(df.columns)
    cols_set = set(cols_list)
    seen = set()
    ordered = []
    for tpl in order:
        if tpl in cols_set and tpl not in seen:
            ordered.append(tpl)
            seen.add(tpl)
            continue
        r = _reduced(tpl)
        if r in cols_set and r not in seen:
            ordered.append(r)
            seen.add(r)
    the_rest = [c for c in cols_list if c not in seen] if keep_rest else []
    key = pd.Index(ordered + the_rest, dtype=object)
    return df.loc[:, key]

def reassing_placeholder_value(experiments):
    def _get(exp, attr, default=None):
        return getattr(exp, attr, default)
    for i in range(len(experiments)):
        if _get(experiments[i], 'pre_smooth') is None:
            experiments[i].pre_smooth = False
        if _get(experiments[i], 'post_smooth') is None:
            experiments[i].post_smooth = False
    return experiments

