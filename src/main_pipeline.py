from data_utils import *
from eval_utils import *
from gpt3_utils import *

import sys
from glob import glob
import json
import shutil
import wandb
import time
import copy
import logging
import math
from tqdm import tqdm


def load_datasets(data_name, task, subset_num=None):
    """
    :param data_name: Filename for Dataset (loads filename + {"train","dev","test"}
    :param task: NER or RE
    :return: (Train, Dev, Test) Tuple with Pandas DataFrames
        NER DataFrame Columns: ['sents', 'ner_seq']
        RE DataFrame Columns: ['text', 'head', 'tail', 'relation']
    """
    data_path = '../data/' + data_name
    assert os.path.exists(
        data_path), 'Create a new directory under `data/` containing the datasets following the conventions shown in the example directory.'

    if task == 'NER':
        return load_ner_dataset(data_path, subset_num)
    elif task == 'RE':
        return load_re_dataset(data_path, subset_num)
    else:
        assert False, print("task parameter must be one of {NER, RE}")


def re_balanced_subset(df, num_samples, seed):
    labels = df['label'].unique()

    permuted_label_specific_dfs = []
    for label in labels:
        label_df = df[df['label'] == label]
        label_df = label_df.sample(len(label_df), random_state=np.random.RandomState(seed))
        permuted_label_specific_dfs.append(label_df)

    subset = []
    curr_sample_per_label = [0 for _ in labels]
    curr_step = 0

    while len(subset) < num_samples:
        curr_label = curr_step % len(labels)

        label_df = permuted_label_specific_dfs[curr_label]
        curr_sample = curr_sample_per_label[curr_label]

        if curr_sample < len(label_df):
            subset.append(label_df[label_df.index == label_df.index[curr_sample]])
            curr_sample_per_label[curr_label] = curr_sample + 1
        else:
            pass

        curr_step += 1

    subset = pd.concat(subset)
    assert len(subset) == num_samples
    return subset


def re_stratified_subset(df, num_samples, seed):
    labels = df['label'].unique()
    label_perc = df.groupby('label').count().sort_values('sents') / len(df)
    num_per_label = [math.ceil(label_perc.loc[label]['sents'] * num_samples) for label in labels]

    permuted_label_specific_dfs = []
    for label in labels:
        label_df = df[df['label'] == label]
        label_df = label_df.sample(len(label_df), random_state=np.random.RandomState(seed))
        permuted_label_specific_dfs.append(label_df)

    subset = []
    curr_sample_per_label = [0 for _ in labels]
    curr_step = 0

    while len(subset) < num_samples and len(subset) < len(df):
        curr_label = curr_step % len(labels)

        label_df = permuted_label_specific_dfs[curr_label]
        curr_sample = curr_sample_per_label[curr_label]

        if curr_sample < num_per_label[curr_label] and curr_sample < len(label_df):
            subset.append(label_df[label_df.index == label_df.index[curr_sample]])
            curr_sample_per_label[curr_label] = curr_sample + 1
        else:
            pass

        curr_step += 1

    subset = pd.concat(subset)
    assert len(subset) == num_samples or len(subset) == len(df)
    return subset

def create_and_save_subset(data_name, task, num_samples, seed, subset_strategy):
    """

    :param data_name: Filename for Dataset
    :param task: NER or RE
    :param num_samples: Number of samples for training subset
    :param seed: Random seed for training dataset sampling
    :param subset_strategy: Strategy for sampling from training set. Natural or Balanced.
    :return: subset_num: Number under which this training subset was saved.
    """
    create_new_subset = True

    # Configuration file
    config = {'data_name': data_name,
              'task': task,
              'num_samples': num_samples,
              'seed': seed,
              'subset_strategy': subset_strategy}

    # Making sure subset directory exists and determining whether this subset was made already.
    subset_dir = '../data/' + data_name + '/training_subsets'

    if not(os.path.exists(subset_dir)):
        os.makedirs(subset_dir)
    else:
        # Checking other subsets for shared configuration
        data_subsets = glob(subset_dir + '/*')
        for other_subset_path in data_subsets:
            other_subset_num = other_subset_path.split('/')[-1]
            f = open(other_subset_path + '/subset_config.{}.json'.format(other_subset_num), "r")
            other_subset_config = json.load(f)
            if other_subset_config == config:
                if os.path.exists(other_subset_path + '/subset.csv'):
                    create_new_subset = False
                subset_num = other_subset_num
                break

    if create_new_subset:
        subset_num = len(glob(subset_dir + '/*'))

        subset_path = subset_dir + '/{}'.format(subset_num)
        assert not (os.path.exists(
            subset_path)), 'Directory structure has been compromised. Subsets should not be added manually.'
        os.makedirs(subset_path)

        f = open(subset_path + '/subset_config.{}.json'.format(subset_num), "w")
        json.dump(config, f)
        f.close()

        # Loading Main Datasets
        print('Loading Dataset.')
        train, dev, test = load_datasets(data_name, task)

        # Sampling Training Set
        print('Creating and Saving Subsets.')
        if subset_strategy == 'natural':
            train_subset = train.sample(num_samples, random_state=np.random.RandomState(seed))
        elif subset_strategy == 'balanced' and task == 'RE':
            train_subset = re_balanced_subset(train, num_samples, seed)
        elif subset_strategy == 'stratified' and task == 'RE':
            train_subset = re_stratified_subset(train, num_samples, seed)
        else:
            assert False, print("{} Sampling Strategy not implemented for task {}.".format(subset_strategy, task))

        # Saving Subset
        train_subset.to_csv(subset_path + '/subset.csv',sep='\t')

        # Saving CONLL subset for NER task
        if task == 'NER':
            write_conll_format_file(subset_path + '/subset.conll.csv', train_subset, 'sents', 'ner_seq')

    return subset_num

def get_config_and_check_duplicate(data_name, subset_num, config_num, plm_or_gpt3='plms'):

    # Download yaml config file
    yaml_config_file_name = '../configs/{}/{}/config.yaml'.format(plm_or_gpt3, config_num)
    yaml_config = load_yaml(yaml_config_file_name)

    # Check if any experiment directory has same hyperparameters and use the same subset number (Return error message)
    output_dir = '../outputs/{}/{}'.format(data_name, plm_or_gpt3)
    experiments = glob(output_dir + '/*')
    for exp in experiments:
        exp_yaml = load_yaml(exp + '/config.yaml')
        if exp_yaml == yaml_config:
            if plm_or_gpt3 == 'gpt3' and os.path.exists(exp + '/subset_config.{}.json'.format(subset_num)):
                #Only when subset is the same can you add more test samples
                experiment_num = int(exp.split('/')[-1])
                output_dir += '/{}'.format(experiment_num)
                return yaml_config, output_dir, experiment_num
            else:
                assert not (os.path.exists(exp + '/subset_config.{}.json'.format(
                    subset_num))), 'Configuration used already for this subset in experiment {}. Check run details.'.format(exp)

    # Create experiment directory
    experiment_num = len(experiments)
    output_dir += '/{}'.format(experiment_num)
    os.makedirs(output_dir)

    # Copying Subset Config
    subset_dir = '../data/' + data_name + '/training_subsets/{}/subset_config.{}.json'.format(subset_num, subset_num)
    shutil.copy(subset_dir, output_dir + '/subset_config.{}.json'.format(subset_num, subset_num))

    yaml.dump(yaml_config, open(output_dir + '/config.yaml','w'))

    return yaml_config, output_dir, experiment_num

def create_sweeps(data_name, subset_num, plm_config_num, grid):
    """
    :param data_name: Filename for Dataset
    :param task: NER or RE
    :param subset_num:
    :param plm_config_num:
    :return: Experiment Number, (CV Sweep Id, Grid Search Sweep Id)
    """

    yaml_config, output_dir, experiment_num = get_config_and_check_duplicate(data_name, subset_num, plm_config_num, 'plms')

    # Add experiment specific parameters like data_name, subset_num and experiment_num
    yaml_config['parameters']['data_name'] = {'values': [data_name]}
    yaml_config['parameters']['subset_num'] = {'values': [subset_num]}
    yaml_config['parameters']['experiment_num'] = {'values': [experiment_num]}

    # Create CV Sweep and Grid Sweep different parameters
    cv_config = copy.deepcopy(yaml_config)
    cv_config['parameters']['run_type'] = {'values': ['cv']}
    cv_config['parameters']['kfolds'] = {'values': [5]}
    cv_config['parameters']['do_predict'] = {'values': [False]}

    grid_config = copy.deepcopy(yaml_config)
    grid_config['parameters']['run_type'] = {'values': ['grid']}
    grid_config['parameters']['kfolds'] = {'values': [1]}
    grid_config['parameters']['do_predict'] = {'values': [True]}

    # Copying YAML
    yaml.dump(cv_config, open(output_dir + '/cv.config.yaml','w'))

    cv_sweep_id = wandb.sweep(cv_config, project=yaml_config['project'])

    if grid:
        yaml.dump(grid_config, open(output_dir + '/grid.config.yaml','w'))
        grid_sweep_id = wandb.sweep(grid_config, project=yaml_config['project'])
    else:
        grid_sweep_id = None

    return experiment_num, (cv_sweep_id, grid_sweep_id)


def run_sweeps(sweeps, available_gpus, grid):

    cv_sweep, grid_sweep = sweeps

    if grid:
        if available_gpus > 1:
            cv_gpus, grid_gpus = np.array_split(available_gpus, 2)

            run_sweep(cv_sweep, cv_gpus, 'cv')
            run_sweep(grid_sweep, grid_gpus, 'grid')
        else:
            print('Unable to run grid as well with only one GPU available')
            run_sweep(cv_sweep, available_gpus, 'cv')
    else:
        run_sweep(cv_sweep, available_gpus, 'cv')

def run_sweep(sweep, available_gpus, outfile):
    for gpu in available_gpus:
        os.system("CUDA_VISIBLE_DEVICES={} wandb agent {} &>> {}_std.txt".format(gpu, sweep, outfile))

def wait_for_sweeps(data_name, experiment_num, sweeps, sweep_types):

    time.sleep(3)

    sweep_times = {}
    start_time = time.time()
    while True:
        for sweep, sweep_type in zip(sweeps, sweep_types):

            #Check Sweep State
            state = wandb.api.get_sweep_state(sweep)
            if state == 'FINISHED':

                #Check Run State
                output_dir = '../outputs/{}/plms/{}/{}.{}'.format(data_name, experiment_num, sweep_type, sweep)

                all_finished = False
                runs = glob(output_dir+'/*')
                for run_path in runs:
                    run_id = run_path.split('/')[-1]
                    api = wandb.Api()
                    run_info = api.run(wandb.run.entity+'/'+wandb.run.project+'/'+run_id)
                    run_state = run_info.state
                    if run_state != 'finished':
                         all_finished = False
                         break
                    else:
                        all_finished = True

                done = all_finished
            else:
                done = False

            if done:
                sweep_times[sweep + '|||' + sweep_type] = time.time() - start_time

        if len(sweep_times) == len(sweeps):
            break
        else:
            time.sleep(20)

    return sweep_times

def get_results(output_dir, parameters, exp_type, dataset_type='dev', sort_by='f1'):
    # Get best CV parameters

    if os.path.exists(output_dir+'/cv/dev.metrics'):
        table = pd.read_csv(output_dir+'/cv/dev.metrics')
    else:
        runs = glob(output_dir + '/{}*/*'.format(exp_type))

        table = []
        for run in runs:
            table.append(pd.read_csv(run + '/{}.metrics'.format(dataset_type)))
        table = pd.concat(table)

    best_params = {}
    sorted_vals = table.sort_values(sort_by, ascending=False).reset_index()

    for param in parameters:
        param_value = sorted_vals.loc[0, param]

        try:
            param_value = eval(param_value)
        except:
            pass

        best_params[param] = param_value

    return table, best_params

def get_unique_params(config):

    best_params = {}
    parameters = config['eval_params']

    for param in parameters:
        param_values = config[param]
        assert len(param_values) == 1, ipdb.set_trace()
        param_value = param_values[0]

        try:
            param_value = eval(param_value)
        except:
            pass


        best_params[param] = param_value

    return best_params

def run_best_cv(yaml_config, output_dir, cv_params, parameters):
    grid_config = copy.deepcopy(yaml_config)
    grid_config['parameters']['run_type'] = {'values': ['grid']}
    grid_config['parameters']['kfolds'] = {'values': [1]}
    grid_config['parameters']['do_predict'] = {'values': [True]}

    for param in parameters:
        grid_config['parameters'][param] = {'values':[float(cv_params[param])]}

    try:
        grid_config['parameters']['per_device_train_batch_size'] = grid_config['parameters']['batch_size']
    except:
        pass

    yaml.dump(grid_config, open(output_dir + '/grid.config.yaml', 'w'))

    grid_sweep_id = wandb.sweep(grid_config, project=yaml_config['project'])
    run_sweep(grid_sweep_id, [wandb.config.available_gpus[0]], 'grid')

    return grid_sweep_id

def synthesize_plm_results(data_name, experiment_num, times, grid):
    output_dir = '../outputs/{}/plms/{}'.format(data_name, experiment_num)

    yaml_config_file_name = output_dir + '/cv.config.yaml'
    yaml_config = load_yaml(yaml_config_file_name)
    parameters = yaml_config['parameters']['eval_params']['values'][0]

    # Get best CV search parameters
    cv_table, cv_params = get_results(output_dir, parameters, 'cv')

    if not(grid):
        grid_sweep_id = run_best_cv(yaml_config, output_dir, cv_params, parameters)
        wait_for_sweeps(data_name, experiment_num, [grid_sweep_id], ['grid'])

    # Get best Dev Grid search parameters
    dev_grid_table, grid_params = get_results(output_dir, parameters, 'grid')
    test_grid_table, _ = get_results(output_dir, parameters, 'grid', dataset_type='test')

    # Create main table
    dev_cv_best = filter_by_dict(dev_grid_table, cv_params)
    test_cv_best = filter_by_dict(test_grid_table, cv_params)

    dev_grid_mean = dev_grid_table.mean().to_frame().transpose()
    dev_grid_best = filter_by_dict(dev_grid_table, grid_params)
    test_grid_best = filter_by_dict(test_grid_table, grid_params)

    cv_table.to_csv(output_dir + '/cv_all_results.csv')
    dev_grid_table.to_csv(output_dir + '/grid_all_results.csv')

    main_table = pd.concat([dev_grid_mean, dev_cv_best, dev_grid_best, test_cv_best, test_grid_best])
    main_table.index = ['Mean Dev', 'Best CV Dev', 'Best Grid Dev', 'Best CV Test', 'Best Grid Test']
    main_table.to_csv(output_dir + '/main_table.csv')

    #Save Times
    # json.dump(times, open(output_dir+'/time_taken.json','w'))

def augment_dataframe(df, task, yaml_config):

    if task == 'NER':
        df = augment_bio_dataframe(df, yaml_config)
    elif task == 'RE':
        df = augment_re_dataframe(df, yaml_config)
    else:
        assert False, print("task parameter must be one of {NER, RE}")

    return df


def run_gpt3(df, task, run_params):
    if task == 'NER':
        result_df = run_gpt3_ner_df(df, run_params)
    elif task == 'RE':
        result_df = run_gpt3_re_df(df, run_params)

    return result_df

def run_gpt3_cv(data_name, task, subset_num, gpt3_config_num):
    #Loading Subset
    subset_train, dev, test = load_datasets(data_name, task, subset_num)

    #Getting Experiment Configuration
    yaml_config, output_dir, experiment_num = get_config_and_check_duplicate(data_name, subset_num, gpt3_config_num, 'gpt3')

    print('Adding Data to GPT-3 Experiment #{} for Dataset {}'.format(experiment_num, data_name))

    if 'ablation' in yaml_config or 'test_only' in yaml_config:
        return experiment_num

    #Looping through Parameters
    curr_run_list = [{}]
    new_run_list = []
    for parameter in yaml_config['eval_params']:
        for run in curr_run_list:
            for value in yaml_config[parameter]:
                new_run = copy.deepcopy(run)
                new_run.update({parameter:value})
                new_run_list.append(new_run)
        curr_run_list = new_run_list
        new_run_list = []

    for i in range(len(curr_run_list)):
        yaml_config.update(curr_run_list[i])
        curr_run_list[i] = copy.deepcopy(yaml_config)

    #Make CV Directory, Create Prompt Dataset and Run GPT-3 Cross-Validation
    if yaml_config['fine_tuning']:
        if 'fine_tuned_model_name' in yaml_config:
            print('Fine Tuned Exists.')
            ipdb.set_trace()
        else:
            print("Fine Tuning...")

            assert len(curr_run_list) == 1
            run_params = curr_run_list[0]
            original_subset_train = subset_train
            subset_train = copy.deepcopy(original_subset_train)

            print('Fine Tuning GPT-3 with the following parameter config:')
            print(run_params)
            print('\n\n')

            #Saving Run Parameters
            json.dump(run_params, open(output_dir+'/fine_tune_params.json','w'))

            #Creating and Saving Prompt Dataset
            subset_train = augment_dataframe(subset_train, task, run_params)
            subset_train.to_csv(output_dir+'/subset.gpt3.csv',sep='\t')

            if task == 'RE':
                training_file = subset_train[['empty_prompts','verbalized_label']]
            elif task == 'NER':
                empty_prompt_lengths = [len(p) for p in subset_train.empty_prompts]
                completions = [p[l + 1:] for p, l in zip(subset_train.prompts, empty_prompt_lengths)]
                subset_train['completion'] = [e if e.strip() != '' else 'None' for e in completions]
                training_file = subset_train[['empty_prompts', 'completion']]

            training_file.columns = ['prompt','completion']
            train_file_name = '_'.join(output_dir.split('/')[2:])+'_gpt3_fine_tuning_file.csv'
            training_file.to_csv(output_dir+'/'+train_file_name)

            os.chdir(output_dir)
            os.system("openai tools fine_tunes.prepare_data -f {}".format(train_file_name))
            ipdb.set_trace()
            if task == 'NER':
                os.system('openai api fine_tunes.create -t "{}_prepared.jsonl" -m {}'.format(train_file_name.split('.')[0],run_params['model']))
            else:
                os.system('openai api fine_tunes.create -t "{}_prepared.jsonl" -m {}'.format(train_file_name.split('.')[0],run_params['model']))
            ipdb.set_trace()
            print("Wait for model to fine tune and re-run with same configuration after adding the model name to the config file.")
            sys.exit()

    else:
        cv_output_dir = output_dir + '/cv'

        if os.path.exists(cv_output_dir + '/dev.metrics'):
            print('CV Already Done.')
        else:
            if os.path.exists(cv_output_dir):
                shutil.rmtree(cv_output_dir)

            os.makedirs(cv_output_dir)

            original_subset_train = subset_train

            cv_metrics = []
            for i,run_params in enumerate(curr_run_list):
                subset_train = copy.deepcopy(original_subset_train)

                run_dir = cv_output_dir+'/{}'.format(i)
                os.makedirs(run_dir)
                print('Running GPT-3 LOOCV for the following parameter config:')
                print(run_params)
                print('\n\n')

                #Saving Run Parameters
                json.dump(run_params, open(run_dir+'/params.json','w'))

                #Creating and Saving Prompt Dataset
                subset_train = augment_dataframe(subset_train, task, run_params)
                subset_train.to_csv(run_dir+'/subset.gpt3.csv',sep='\t')

                cv_dev = get_prompts_from_df(subset_train, None, run_params['in_context_size'], run_params['sampling_strategy'], random_seed=run_params['random_seed'], cross_val=True)

                cv_output = run_gpt3(cv_dev, task, run_params)
                cv_output.to_csv(run_dir+'/gpt3.output.temp.csv',sep='\t')

                metric_dict = evaluate_gpt3_output(cv_output, task, run_params)
                metric_dict.update(run_params)
                pd.DataFrame([metric_dict]).to_csv(run_dir + '/dev.metrics')
                cv_metrics.append(metric_dict)

                cv_output.to_csv(run_dir+'/gpt3.output.csv',sep='\t')
                os.remove(run_dir+'/gpt3.output.temp.csv')

            cv_metrics = pd.DataFrame(cv_metrics)
            cv_metrics.to_csv(cv_output_dir+'/dev.metrics')

    return experiment_num


def get_stratified_test_set(test_df, num_samples):
    #Stratified Sampling for Relation Classification
    test_df = re_stratified_subset(test_df, num_samples, 42)
    return test_df

def run_gpt3_final(data_name, task, subset_num, num_test_samples, gpt3_experiment_num):

    subset_train, dev, full_test = load_datasets(data_name, task, subset_num)

    output_dir = '../outputs/{}/{}/{}'.format(data_name, 'gpt3',gpt3_experiment_num)
    yaml_config = load_yaml(output_dir + '/config.yaml')

    if 'ablation' in yaml_config or 'test_only' in yaml_config or yaml_config['fine_tuning'][0]:
        cv_params = get_unique_params(yaml_config)

        if 'ablation' in yaml_config:
            full_test = dev
    else:
        cv_metrics, cv_params = get_results(output_dir, yaml_config['eval_params'], 'cv')
        cv_summary = cv_metrics[['total_cost', 'total_time']].sum()
        cv_summary.to_csv(output_dir + '/dev.cost.summary')

    subset_train = augment_dataframe(subset_train, task, cv_params)

    prev_test = None
    if os.path.exists(output_dir + '/test.cv_best.gpt3.output.csv'):

        if task == 'NER':
            prev_test = pd.read_csv(output_dir + '/test.cv_best.gpt3.output.csv', sep='\t', index_col=0)
            prev_test.entities = [eval(p) for p in prev_test.entities]
            prev_test.predictions = [eval(p) for p in prev_test.predictions]
        else:
            prev_test = pd.read_csv(output_dir + '/test.cv_best.gpt3.output.csv', sep='\t')

        prev_test_size = len(prev_test)

        shutil.copy(output_dir + '/test.cv_best.gpt3.output.csv', output_dir + '/test.cv_best.gpt3.output.{}.csv'.format(prev_test_size))

        if num_test_samples <= prev_test_size:
            assert False, print("GPT-3 Experiment Done. Try more test samples or a different configuration.")
        else:
            if task == 'RE':
                test = get_stratified_test_set(full_test, num_test_samples)
                #Remove previously run examples (RE datasets must have 'id' column
                test = test[[i not in prev_test.id.values for i in test.id.values]]
            else:
                test = full_test.sample(len(full_test), random_state=np.random.RandomState(seed=42))
                test = test[prev_test_size:num_test_samples]
    else:
        if task == 'RE':
            test = get_stratified_test_set(full_test, num_test_samples)
        else:
            test = full_test.sample(len(full_test), random_state=np.random.RandomState(seed=42))
            test = test[:num_test_samples]

    test = augment_dataframe(test, task, cv_params)

    subset_train.to_csv(output_dir + '/subset.cv_best.gpt3.csv', sep='\t')
    test.to_csv(output_dir + '/test.cv_best.gpt3.csv', sep='\t')

    best_test = get_prompts_from_df(subset_train, test, cv_params['in_context_size'], cv_params['sampling_strategy'],
                                 random_seed=cv_params['random_seed'],
                                 cross_val=False)

    if os.path.exists(output_dir + '/test.cv_best.gpt3.output.temp.csv'):
        best_test_output = pd.read_csv(output_dir + '/test.cv_best.gpt3.output.temp.csv', sep='\t', index_col=0)
        #Turning strings into lists for further evaluation
        if task == 'NER':
            best_test_output.entities = [eval(p) for p in best_test_output.entities]
            best_test_output.predictions = [eval(p) for p in best_test_output.predictions]
    else:
        best_test_rows = []
        print('Running GPT-3 on {} Examples.'.format(len(best_test)))
        for i, row in tqdm(best_test.iterrows()):
            df_row = row.to_frame().T
            row_output = run_gpt3(df_row, task, cv_params)
            best_test_rows.append(row_output)

            best_test_output = pd.concat(best_test_rows)
            best_test_output.to_csv(output_dir + '/test.cv_best.gpt3.output.temp.csv', sep='\t')

    if prev_test is not None:
        best_test_output = pd.concat([prev_test, best_test_output])

    metric_dict = evaluate_gpt3_output(best_test_output, task, cv_params)
    metric_dict.update(cv_params)
    pd.DataFrame([metric_dict]).to_csv(output_dir + '/test.metrics')

    best_test_output.to_csv(output_dir + '/test.cv_best.gpt3.output.csv', sep='\t')
    os.remove(output_dir + '/test.cv_best.gpt3.output.temp.csv')

hyperparameter_defaults = dict(
    create_subset = True,
    data_name = 'BC5CDR-disease',
    task = 'NER',
    num_samples = 100,
    subset_seed = 42,
    subset_strategy = 'natural',
    subset_num = None,
    num_test_samples = 250,
    available_gpus = [3],
    test_plms = False,
    evaluate_plms = False,
    grid = False,
    plm_config_num = 0,

    test_gpt3 = True,
    gpt3_config_num = 0
)

def main():

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.FATAL)
    wandb.util.logging.disable()

    create_subset = wandb.config.create_subset
    data_name = wandb.config.data_name
    task = wandb.config.task
    num_samples = wandb.config.num_samples
    subset_seed = wandb.config.subset_seed
    subset_strategy = wandb.config.subset_strategy
    subset_num = wandb.config.subset_num
    num_test_samples = wandb.config.num_test_samples
    available_gpus = wandb.config.available_gpus
    test_plms = wandb.config.test_plms
    evaluate_plms = wandb.config.evaluate_plms
    plm_config_num = wandb.config.plm_config_num

    test_gpt3 = wandb.config.test_gpt3
    gpt3_config_num = wandb.config.gpt3_config_num
    grid = wandb.config.grid

    print(wandb.config)
    if create_subset:
        print('Creating Subset...')
        subset_num = create_and_save_subset(data_name, task, num_samples, subset_seed, subset_strategy)
        print('Done Creating Subset.')

    if test_plms:
        # Run 2 sweeps with as many GPUs as available up to a threshold

        print('Testing PLMs...')
        # Make sure the output of the following functions follow the directory structure and create the outputs we want
        experiment_num, sweeps = create_sweeps(data_name, subset_num,
                                               plm_config_num, grid)  # K-fold CV and Grid Search Sweeps

        run_sweeps(sweeps, available_gpus, grid)

        try:
            if grid:
                times = wait_for_sweeps(data_name, experiment_num, sweeps, ['cv','grid'])
            else:
                times = wait_for_sweeps(data_name, experiment_num, [sweeps[0]], ['cv'])
        except:
            for sweep in sweeps:
                wandb.api.cancel_sweep(sweep)
            assert False, 'User Stopping Main Run. Stopped Sweeps.'

    if evaluate_plms:
        if not(test_plms):
            experiment_num = wandb.config.experiment_num
            times = None

        synthesize_plm_results(data_name, experiment_num, times, grid)
        print('Done with PLM Experiment.')

    if test_gpt3:
        print('Running GPT-3 Experiment...')
        gpt3_experiment_num = run_gpt3_cv(data_name, task, subset_num, gpt3_config_num)
        run_gpt3_final(data_name, task, subset_num, num_test_samples, gpt3_experiment_num)

if __name__ == "__main__":
    wandb.init(config=hyperparameter_defaults, project="few-shot-bioIE")

    main()
