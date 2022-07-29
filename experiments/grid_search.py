from sklearn.model_selection import ParameterGrid
import os

from experiments.constants import R_SQUARED
from experiments.synthetic.constants import *
from experiments.synthetic.plot import *


def get_grid_search_space():
    return [{
        'max_lr': [1e-5],
        'epochs': [1],
        'batch': [ 200],
        'pct_start': [0.45],
        'anneal_strategy': ['linear'],
        'base_momentum': [0.85],
        'max_momentum': [0.95],
        'div_factor': [10, 5],
        'final_div_factor': [1e4],
        'weight_decay': [0]
    }]
    # return [{
    #     'max_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    #     'epochs': [10, 20],
    #     'batch': [100, 200],
    #     'pct_start': [0.45],
    #     'anneal_strategy': ['linear'],
    #     'base_momentum': [0.85],
    #     'max_momentum': [0.95],
    #     'div_factor': [10, 5, 2],
    #     'final_div_factor': [1e4],
    #     'weight_decay': [0]
    # }]

"""# Grid Search Setup"""

def grid_search(root_folder, dataset_train, dataset_val, bound_min, bound_max, grid_config, train_callback, checkpoint_name, nb_iterations = 1, conf_validation = None):
    configs = ParameterGrid(grid_config)
    configs_len = len(configs)
    counter = 0
    checkpoint_file = root_folder + '/' + checkpoint_name + '.tar'
    grid_checkpoint_file = root_folder + '/grid ' + checkpoint_name + '.tar'
    try:
        resume_grid_search = t.load(root_folder + '/' + GRID_RESULTS_FILE)
    except FileNotFoundError:
        resume_grid_search = None

    results = {}
    best = [math.inf, math.inf, -math.inf]
    if resume_grid_search is not None and 'best' in resume_grid_search:
        best_conf = resume_grid_search['best']
        print('Best previous configuration', best_conf)
        best = resume_grid_search[str(best_conf)]
        print(f'Best previous metrics abs err = {best[ABS_ERR]}, R2 = {best[R_SQUARED]}')
        results = resume_grid_search

    for conf in ParameterGrid(grid_config):
        counter += 1

        if resume_grid_search is not None and str(conf) in resume_grid_search:
            print('Allready evaluated configuration', conf)
            continue

        if not conf_validation(conf):
            print('Skipping over configuration', conf)
            results[str(conf)] = 'invalid'
            continue

        print('-' * 5, 'grid search {}/{}'.format(counter, configs_len), '-' * 5)
        print('Config:', conf)

        best_from_iterations = [math.inf, math.inf, -math.inf]

        for i in range(nb_iterations):
            if nb_iterations != 1:
                print('Iteration', i + 1)
            metrics = train_callback(dataset_train, dataset_val, bound_min, bound_max, conf)

            # if metrics[R_SQUARED] > best[R_SQUARED]:
            if metrics[ABS_ERR] < best[ABS_ERR] and not (math.isnan(metrics[LOSS] or math.isnan(metrics[ABS_ERR]) or math.isnan(metrics[R_SQUARED]))):
                best_from_iterations = metrics
                best = metrics
                results['best'] = conf
                if os.path.exists(grid_checkpoint_file):
                    os.remove(grid_checkpoint_file)
                os.rename(checkpoint_file, grid_checkpoint_file)
        else:
            results[str(conf)] = best_from_iterations
            t.save(results, root_folder + '/' + GRID_RESULTS_FILE)

    return best


def config_validation(conf):
    return conf['div_factor'] <= conf['final_div_factor'] and conf['max_momentum'] >= conf['base_momentum']