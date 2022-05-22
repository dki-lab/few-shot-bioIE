from main_pipeline import *
import wandb

default_params = {'create_subset': True,
                  'subset_seed': 42,
                  'subset_num': None,
                  'available_gpus': [3],
                  'data_name': 'BC5CDR-disease',
                  'task': 'NER',
                  'subset_strategy': 'natural',
                  'num_samples': 100
                  }

wandb_configs = [{'num_test_samples':50,
                  'test_plms':False,
                  'evaluate_plms': False,
                  'test_gpt3': True,
                  'gpt3_config_num': 0
                  },
                 {'test_plms': True,
                  'evaluate_plms': True,
                  'test_gpt3': False,
                  'plm_config_num': 0
                  }
                 ]

for curr_config in wandb_configs:

    curr_config.update(default_params)

    new_params = hyperparameter_defaults.copy()

    for new_config_param, new_config_value in curr_config.items():
        new_params[new_config_param] = new_config_value

    print(new_params)
    wandb.init(config=new_params, project="few-shot-bioIE")
    print(wandb.config)

    main()

    wandb.finish()