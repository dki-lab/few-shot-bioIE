"""
This script runs GPT-3 in-context learning benchmarking for all NER and RE BLURB datasets with their designated prompt
selection configurations.
Change subset seed to evaluate on different training subsets.

IMPORTANT: All configuration files are set to run the 'ada' version of GPT-3 to avoid accidental running of
expensive model. To benchmark 175B parameter GPT-3 change ada in all configuration files to 'davinci'.
"""

from main_pipeline import *
import wandb

num_test_samples = 1000
subset_seed = 42
hyperparameter_defaults['num_samples'] = 100
hyperparameter_defaults['test_gpt3'] = True
hyperparameter_defaults['test_plms'] = False
hyperparameter_defaults['evaluate_plms'] = False
hyperparameter_defaults['subset_seed'] = subset_seed

wandb_configs = [{'data_name': 'BC5CDR-disease',
                  'task': 'NER',
                  'subset_strategy': 'natural',
                  'gpt3_config_num': 2,
                  'num_test_samples': num_test_samples},
                 {'data_name': 'BC5CDR-chem',
                 'task': 'NER',
                 'subset_strategy': 'natural',
                 'gpt3_config_num': 3,
                 'num_test_samples': num_test_samples},
                 {'data_name': 'BC2GM',
                 'task': 'NER',
                 'subset_strategy': 'natural',
                 'gpt3_config_num': 4,
                 'num_test_samples': num_test_samples},
                 {'data_name': 'NCBI-disease',
                  'task': 'NER',
                  'subset_strategy': 'natural',
                  'gpt3_config_num': 5,
                  'num_test_samples': num_test_samples},
                 {'data_name': 'JNLPBA',
                  'task': 'NER',
                  'subset_strategy': 'natural',
                  'gpt3_config_num': 6,
                  'num_test_samples': num_test_samples},
                 {'data_name': 'DDI',
                  'task': 'DDI',
                  'subset_strategy': 'balanced',
                  'gpt3_config_num': 7,
                  'num_test_samples': num_test_samples},
                 {'data_name': 'chemprot',
                  'task': 'RE',
                  'subset_strategy': 'balanced',
                  'gpt3_config_num': 8,
                  'num_test_samples': num_test_samples},
                 {'data_name': 'gad',
                  'task': 'RE',
                  'subset_strategy': 'balanced',
                  'gpt3_config_num': 9,
                  'num_test_samples': num_test_samples}
                 ]

for config in wandb_configs:

    new_params = hyperparameter_defaults.copy()

    for new_config_param, new_config_value in config.items():
        new_params[new_config_param] = new_config_value

    wandb.init(config=new_params, project="few-shot-bioIE")
    config = wandb.config

    print(config)
    main()
    wandb.finish()