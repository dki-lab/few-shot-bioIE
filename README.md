# True Few-Shot BioIE: GPT-3 In-Context vs. Small PLM Fine-Tuning

This repository provides the pipeline used in [our work](https://arxiv.org/abs/2203.08410) to benchmark 
GPT-3 in-context learning and BERT-sized model fine-tuning on biomedical information extraction tasks (NER and relation extraction) 
under the true few-shot setting. 

We run GPT-3 through the OpenAI API and use the HuggingFace library 
to fine-tune small PLMs.

## Installation

Run the following commands to create a conda environment with the required packages. 

```
conda create -n few-shot-bioIE python=3.9 pip
conda activate few-shot-bioIE
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data

Data for our experiments was obtained from the [BLURB Benchmark](https://microsoft.github.io/BLURB/index.html). To 
download and process the necessary datasets we use a modified version of their data processing scripts.

To download all IE datasets in BLURB, we first run the following script:

```
bash download_BLURB_data.sh
```

This should create the `raw_data` directory under root containing all the raw datasets except for ChemProt. To download ChemProt, please register and download from [this link.](https://biocreative.bioinformatics.udel.edu/resources/corpora/chemprot-corpus-biocreative-vi/) as described in BLURB.
Place the file `ChemProt_Corpus.zip` under the `raw_data` directory and run the following preprocessing script:

```
bash preprocess_BLURB_data.sh
```

This should yield a training, dev and test set for 5 NER datasets (`BC5CDR-disease`,`BC5CDR-chem`,`BC2GM`,`JNLPBA` and `NCBI-disease`) and 3 RE datasets (`DDI`, `ChemProt` and `GAD`).

In order to include new datasets into this system, check the format of the NER and RE datasets after this step. Following
the data format in any of the dataset specific directories should be all that is needed. 

## Running Experiments

### Understanding Configurations

Under the `configs` directory, we find directories for `plms` and `gpt3`, each of these containing .yaml configuration 
files intended for use with the WandB library. We have included one example configuration for each (task, method) 
combination `configs/plms/0` and `configs/gpt3/0` are NER specific configuration files while `configs/plms/1` and 
`configs/gpt3/1` are RE specific. We also include all configuration files necessary for reproducibility which 
we discuss in a later section. For further details about all hyperparameters tested please refer to
[our paper](https://arxiv.org/abs/2203.08410). 

The configuration files `configs/sample_ner_config.yaml` and `configs/sample_re_config.yaml` contain the settings necessary 
to run the main pipeline which manages 1) the creation of small training datasets, 2) the PLM fine-tuning true-shot hyperparameter 
search and 3) the true-few shot prompt selection process for GPT-3. This configuration 
file is most useful for running this process for many PLM fine-tuning models or datasets since in our work we evaluate 
multiple small PLMs. Configurations which have already been run with some training dataset will not be re-run to 
prevent wasted resources.

The following is the configuration directory structure for reference:

```
configs
    plms
        {config_num}
            config.yaml (task specific script,
                        all hyperparameters)
    gpt3
        {config_num}
            config.yaml (dataset_name,
                        model_name, 
                        overall_instructions,
                        sent_intro,
                        retrieval_message,
                        sampling strategy (knn module, random (seed)), 
                        context size)
    sample_ner_config.yaml
    sample_re_config.yaml
    main_ner_config_plms.yaml (PLM Hyperparameter Search for NER in our paper)
    main_re_config_plms.yaml (PLM Hyperparameter Search for RE in our paper)
```

To run an experiment using the example configuration, make sure to look over the main pipeline 
configuration files and understand each parameter set.

### Running the Main Pipeline

Running all our scripts requires WandB. Be sure to login to WandB using `wandb login` and follow their instructions.

There are two main ways to run our main pipeline.
1) If all parameter choices in a specific configuration can be tested with all others 
we leverage the WandB hyperparameter sweep procedure. 
2) On the other hand, if not all parameter combinations should be tested, 
we loop over a set list of specified configurations and run them individually.

To make sure everything is in working order, we recommend starting by running the sample configurations with
the following commands (be sure to specify which GPUs are available in your system):

```
cd src
GPUS={Comma-separated list of available GPUs}
CUDA_VISIBLE_DEVICES=$GPUS python run_sample_configs_ner.py
CUDA_VISIBLE_DEVICES=$GPUS python run_sample_configs_re.py
```

To confirm that the sample scripts run smoothly, output directories for `BC5CDR-disease` and `DDI` should be created 
under the `outputs` directory. Refer to the `Output Structure` for more details on the expected output.

To test that the WandB hyperparameter sweep procedure is also working properly, delete the directories created under the 
`outputs` directory and run the following command (be sure to edit the .yaml file to specify which GPUs are available in 
your system):

```
wandb sweep ../configs/sample_ner_config.yaml
```

The previous command will create a sweep and output a sweep name of the form
`{user_name}/{project_name}/{sweep_id}`. This sweep name must be used to run a 
WandB agent as follows:

```
CUDA_VISIBLE_DEVICES=$GPUS wandb agent {user_name}/{project_name}/{sweep_id}
```

The same output files should be created under `outputs/BC5CDR-disease` as in the previous section.
Any .yaml configuration file can be used in the way just described.

### Few-Shot Benchmarking for BLURB IE (Reproducibility)  

To reproduce the PLM fine-tuning results presented in our paper, run through the
previous WandB procedure with the .yaml configuration files `configs/main_ner_config_plms.yaml` 
and `configs/main_re_config_plms.yaml`. The hyperparameter choices used can be found both in our paper
and in the configuration files under `configs/plms/` (NER hyperparameters from `2` to `6` and RE 
hyperparameters from `7` to `11`).  

To reproduce our true-few shot benchmarking of the 175B GPT-3 model, 
be sure to first change the `model` field in the configuration files under
`configs/gpt3/` from `2` to `9` from `ada` to `davinci` (Note that running the 
largest GPT-3 model can be quite expensive). To carry out the benchmarking run 
the following script:

`CUDA_VISIBLE_DEVICES=$GPUS python benchmarking_gpt3_in_context.py`

### Output Structure

```
outputs
    data_name
        plms
            experiment_num (new number every time)
                config.yaml
                subset_config.{subset_num}.json (copy of the subset dataset configuration)  
                cv.config.yaml
                grid.config.yaml
                                                        
                cv.{sweep_id}
                    {run_id}
                        cv.params.p
                        fold_id
                            all_results.json
                            train.tokens.{epoch}.txt
                            train.predictions.{epoch}.txt
                            train.labels.{epoch}.txt
                            train.metrics.{epoch}_results.json
                            dev.tokens.{epoch}.txt
                            dev.predictions.{epoch}.txt
                            dev.labels.{epoch}_results.json
                            dev.metrics.{epoch}_results.json
                        dev.tokens.{epoch}.txt
                        dev.labels.{epoch}.txt
                        dev.predictions.{epoch}.txt
                        dev.metrics
                        
                grid.{sweep_id}
                    {run_id}
                        grid.params.p
                        train.tokens.{epoch}.txt
                        train.predictions.{epoch}.txt
                        train.labels.{epoch}.txt
                        
                        dev.tokens.{epoch}.txt
                        dev.labels.{epoch}.txt
                        dev.predictions.{epoch}.txt

                        train.metrics.{epoch}_results.json                        
                        dev.metrics.{epoch}_results.json
                        test.metrics.{epoch}_results.json
                        
                        dev.metrics
                        test.metrics

                cv_all_results.csv
                grid_all_results.csv
                main_table.csv
        gpt-3
            experiment_num (new number every time)
                subset_config.{subset_num}.json (copy of the subset dataset configuration)
                config.yaml (subset_num,
                            model_name, 
                            fine_tuning,
                            overall_instructions,
                            sent_intro,
                            retrieval_message,
                            sampling strategy (knn module, random (seed)), 
                            context size)
                                        
                * (Don't re-run anything in GPT-3, all outputs must be recycled)
                
                cv (Directory containing DataFrames for every HP configuration)
                    run_num
                        params.json
                        subset.gpt3.csv
                        gpt3.output.csv
                        dev.metrics
                    dev.metrics

                subset.cv_best.gpt3.csv (Training Dataframe)
                test.cv_best.gpt3.csv (Test Dataframe)
                test.cv_best.gpt3.output.csv (GPT-3 Outputs)
                test.metrics (All Test Results (Best & Worse)
                dev.cost.summary (Cost and compute time)
```

## Testing GPT-3 In-Context Learning Alone

Two Jupyter notebooks under the `src` directory can be used directly to test GPT-3 on manually designed prompts for 
NER and RE tasks. Use these only after running the example configurations above since some configuration files
used in these scripts are created by those first runs. The Jupyter notebooks are the following:

```
src/GPT-3 NER Run Script.ipynb
src/GPT-3 RE Run Script.ipynb
```



