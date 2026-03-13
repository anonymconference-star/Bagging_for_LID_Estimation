from Bagging_for_LID.run_files.geom_prog import *
from Bagging_for_LID.Datasets.dataset_collections import *
#-----------------------------------------------------------------------------------------------------------------------
def param_dicts_general(base_param_dict, variants_test_types, estimator_names, changing_vars, test_name='test'):
    non_bagging_vars = ['sr', 'Nbag', 't', 'pre_smooth', 'submethod_error', 'submethod_0']
    non_simple_bagging_vars = ['t', 'submethod_error', 'submethod_0']
    param_dicts_dict = {}
    for estimator_name in estimator_names:
        for test_type in variants_test_types:
            if test_type == 'weight':
                param_dict = base_param_dict.copy()
                param_dict['estimator_name'] = estimator_name
                if estimator_name == 'tle':
                    param_dict['bagging_method'] = [None, 'bag', 'bagw']
                else:
                    param_dict['bagging_method'] = [None, 'bag', 'bagw', 'bagwth']
            elif test_type == 'smooth':
                param_dict1 = base_param_dict.copy()
                param_dict1['bagging_method'] = None
                param_dict1['pre_smooth'] = False
                param_dict1['post_smooth'] = [True, False]
                param_dict1['estimator_name'] = estimator_name

                param_dict2 = base_param_dict.copy()
                param_dict2['bagging_method'] = 'bag'
                param_dict2['pre_smooth'] = [True, False]
                param_dict2['post_smooth'] = [True, False]
                param_dict2['estimator_name'] = estimator_name

                param_dict = [param_dict1, param_dict2]
            elif test_type == 'variable':
                param_dict1 = base_param_dict.copy()
                param_dict1['estimator_name'] = estimator_name
                param_dict1['bagging_method'] = None

                param_dict2 = base_param_dict.copy()
                param_dict2['estimator_name'] = estimator_name
                param_dict2['bagging_method'] = 'bag'

                param_dict = [param_dict1, param_dict2]
            elif test_type == 'extra_variable':
                param_dict1 = base_param_dict.copy()
                param_dict1['estimator_name'] = estimator_name
                for var in changing_vars:
                    if var in non_bagging_vars:
                        param_dict1[var] = None
                param_dict1['bagging_method'] = None

                param_dict2 = base_param_dict.copy()
                param_dict2['estimator_name'] = estimator_name

                param_dict = [param_dict1, param_dict2]
            elif test_type == 'weight_with_t':
                param_dict0 = base_param_dict.copy()
                param_dict0['estimator_name'] = estimator_name
                param_dict0['bagging_method'] = [None]

                for var in changing_vars:
                    if var in non_bagging_vars:
                        param_dict0[var] = None

                param_dict1 = base_param_dict.copy()
                param_dict1['estimator_name'] = estimator_name
                param_dict1['bagging_method'] = ['bag']
                for var in changing_vars:
                    if var in non_simple_bagging_vars:
                        param_dict1[var] = None

                param_dict2 = base_param_dict.copy()
                param_dict2['estimator_name'] = estimator_name
                if estimator_name == 'tle':
                    param_dict2['bagging_method'] = ['bagw']
                else:
                    param_dict2['bagging_method'] = ['bagw', 'bagwth']

                param_dict = [param_dict0, param_dict1, param_dict2]
            else:
                ValueError('Test type is invalid')
            param_dicts_dict[f'{test_name}_{estimator_name}_{test_type}'] = param_dict
    return param_dicts_dict
#-----------------------------------------------------------------------------------------------------------------------
#dataset_name_strings = ['uniform']
#Just for data generation
param_dicts_data = {'dataset_name': dataset_name_strings,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': 'mle',
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': 10,
                    'sr': 0.3,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 1}
#-----------------------------------------------------------------------------------------------------------------------
#MAIN Paper results

#Effectiveness of bagging/Bagging and smoothing
effectiveness_test_base_param_dict = {'dataset_name': dataset_name_strings,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': None,
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_prog_smoothing_weighing_tests,
                    'sr': sr_prog_smoothing_weighing_tests,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 1}

#effectiveness_variants_test_types = ['weight', 'smooth']
effectiveness_variants_test_types = ['smooth']
#effectiveness_variants_test_types = ['weight']
#effectiveness_estimator_names = ['mle']
effectiveness_estimator_names = ['mle', 'mada', 'tle']

#Variable tests

#Number of bags test (mse bar charts)
Nbag_test_base_param_dict = {'dataset_name': dataset_name_strings,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': None,
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': 10,
                    'sr': 0.05,
                    'Nbag': Nbag_prog_number_of_bags_tests,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 1}

variable_variants_test_types = ['variable']
variable_estimator_names = ['mle', 'mada', 'tle']

#Sampling rate test (mse bar charts)
sr_prog_test_base_param_dict = {'dataset_name': dataset_name_strings,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': None,
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': 10,
                    'sr': sr_prog_sampling_rate_test,
                    'Nbag': 30,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 1}

#Interaction of sampling rate and number of bags (mse difference heatmaps)
interaction_sr_Nbag_test_base_param_dict = {'dataset_name': dataset_name_strings,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': None,
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': 10,
                    'sr': sr_prog_sampling_rate_number_of_bags_interaction,
                    'Nbag': Nbag_prog_sampling_rate_number_of_bags_interaction,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 1}

#Interaction of sampling rate and k (mse difference heatmaps)
interaction_sr_k_test_base_param_dict = {'dataset_name': dataset_name_strings,
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': None,
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_prog_sampling_rate_k_interaction,
                    'sr': sr_prog_sampling_rate_k_interaction,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': 1}

effectiveness_test_with_t_base_param_dict = {'dataset_name': dataset_name_strings[-1],
                    'n': 2500,
                    'lid': None,
                    'dim': None,
                    'estimator_name': None,
                    'bagging_method': None,
                    'submethod_0': '0',
                    'submethod_error': 'log_diff',
                    'k': k_prog_smoothing_weighing_tests,
                    'sr': sr_prog_smoothing_weighing_tests,
                    'Nbag': 10,
                    'pre_smooth': False,
                    'post_smooth': False,
                    't': t_prog_small}





