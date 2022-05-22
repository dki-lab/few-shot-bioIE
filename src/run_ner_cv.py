#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import sys
from custom_callbacks import *

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
import pickle
import json
import time
import gc

from data_utils import *

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.utils.versions import require_version

import ipdb
import torch
import random
import wandb
from glob import glob
from eval_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


hyperparameter_defaults = dict(
    data_name='BC5CDR-disease',
    data_dir='../data/',
    output_dir='../outputs/',
    subset_num=0,
    experiment_num=0,
    run_type='grid',#cv, grid, best, worst
    kfolds=5,
    do_train=True,
    do_eval=True,
    do_predict=True,

    per_device_train_batch_size=16,
    learning_rate=3e-5,
    num_train_epochs=10,
    weight_decay=0.1,
    warmup_ratio=0.06,

    model_name='./disease_pubmedbert',
    per_device_eval_batch_size=16,
    seed=42,
    logging_strategy='no',
    evaluation_strategy='steps',
    save_strategy='no',
    eval_steps_epochs=None,
    save_total_limit=0,
    metric="seqeval",
    max_seq_length=128,
    max_eval_samples=None,
    max_predict_samples=None,
    overwrite_output_dir=True,
    gradient_checkpointing=True,
    disable_tqdm=True,
    epoch_eval_period=5,
    label_all_tokens=True
)

wandb.init(config=hyperparameter_defaults, project="bertNER")
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    dataset_dir = os.path.join(wandb.config.data_dir, wandb.config.data_name)
    wandb.config.training_file = os.path.join(dataset_dir, 'training_subsets/{}/subset.conll.csv'.format(wandb.config.subset_num))
    wandb.config.dev_file = os.path.join(dataset_dir, 'dev.conll.csv')
    wandb.config.test_file = os.path.join(dataset_dir, 'test.conll.csv')

    wandb_dict = dict(wandb.config)
    #Follow directory structure for output directory under WandB Sweep and Run ids
    output_dir = '/'.join([wandb.config.output_dir,
                          wandb.config.data_name,
                          'plms',
                          str(wandb.config.experiment_num),
                          wandb.config.run_type+'.'+str(wandb.run.sweep_id),
                          wandb.run.id])
    os.makedirs(output_dir)
    pickle.dump(dict(wandb.config), open(os.path.join(output_dir, '{}.params.p'.format(wandb.config.run_type)), 'wb'))
    wandb_dict['output_dir'] = output_dir

    #Using WandB arguments to feed into huggingface training arguments
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict(wandb_dict)[0]

    #Setup logging
    logger = logging.getLogger(__name__)
    log_level = logging.FATAL
    logger.setLevel(log_level)

    log_level = transformers.logging.FATAL
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity(log_level)
    transformers.logging.disable_progress_bar()

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.FATAL)
    wandb.util.logging.disable()
    #
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    seed_torch(training_args.seed)

    raw_datasets = load_dataset('NERSingleWandB.py')

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features

    text_column_name = "tokens"
    label_column_name = "ner_tags"

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        wandb.config.model_name,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()}
    )

    tokenizer_name_or_path = wandb.config.model_name
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            add_prefix_space=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length"

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=wandb.config.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if wandb.config.label_all_tokens:
                        if label[word_idx] == 0:
                            label_ids.append(label_to_id[label[word_idx]])
                        else:
                            label_ids.append(label_to_id[2])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    #Loading Test Dataset before k-fold CV
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]

        if wandb.config.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(wandb.config.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                desc="Running tokenizer on prediction dataset",
                load_from_cache_file=False
            )
    else:
        predict_dataset = None

    original_train_dataset = raw_datasets["train"]
    fold_size = int(len(original_train_dataset) / wandb.config.kfolds)

    for fold in range(wandb.config.kfolds):

        if wandb.config.kfolds > 1:
            training_args.output_dir = os.path.join(output_dir, str(fold))
            os.makedirs(training_args.output_dir)

            train_dataset = original_train_dataset.filter(lambda i: int(i) < fold * fold_size or int(i) >= (fold + 1) * fold_size, input_columns=['id'])
            dev_dataset = original_train_dataset.filter(lambda i: int(i) >= fold * fold_size and int(i) < (fold + 1) * fold_size, input_columns=['id'])

            assert len(dev_dataset) == fold_size
            assert len(train_dataset) == len(original_train_dataset) - fold_size
        else:
            train_dataset = raw_datasets['train']
            dev_dataset = raw_datasets['validation']

        if wandb_dict['eval_steps_epochs'] is not None:
            training_args.eval_steps = wandb_dict['eval_steps_epochs'] * (
                        int(len(original_train_dataset) / training_args.per_device_train_batch_size) + 1)

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(function=tokenize_and_align_labels,
                batched=True,
                desc="Running tokenizer on train dataset",
                load_from_cache_file=False
            )

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = dev_dataset.map(function=tokenize_and_align_labels,
                batched=True,
                desc="Running tokenizer on validation dataset",
                load_from_cache_file=False
            )

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer,
                                                           pad_to_multiple_of=8 if training_args.fp16 else None)

        # Metrics
        metric = load_metric("seqeval")


        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        model = AutoModelForTokenClassification.from_pretrained(
            wandb.config.model_name,
            config=config
        )

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.add_callback(EvalReportingCallback(model, trainer, train_dataset, eval_dataset, predict_dataset, label_list))

        # Training
        if training_args.do_train:
            train_result = trainer.train()

        # wandb.log({'training_data_size': len(train_dataset)})

        del model
        del trainer
        time.sleep(2)
        gc.collect()
        time.sleep(2)
        torch.cuda.empty_cache()

    if wandb.config.run_type == 'cv':
        results_by_epoch = []

        #Aggregate results and compute metric
        for epoch in range(wandb.config.num_train_epochs // wandb.config.epoch_eval_period):
            epoch += 1  # Starts at
            epoch *= wandb.config.epoch_eval_period

            tokens = []
            labels = []
            preds = []

            for kfold in range(wandb.config.kfolds):
                tokens.extend(open(output_dir + '/{}/dev.tokens.{}.txt'.format(kfold, epoch), 'r').readlines())
                labels.extend(open(output_dir + '/{}/dev.labels.{}.txt'.format(kfold, epoch), 'r').readlines())
                preds.extend(open(output_dir + '/{}/dev.predictions.{}.txt'.format(kfold, epoch), 'r').readlines())

            tok_file = open(output_dir + '/dev.tokens.{}.txt'.format(epoch), 'w')
            labels_file = open(output_dir + '/dev.labels.{}.txt'.format(epoch), 'w')
            preds_file = open(output_dir + '/dev.predictions.{}.txt'.format(epoch), 'w')

            tok_file.writelines(tokens)
            labels_file.writelines(labels)
            preds_file.writelines(preds)

            tok_file.close()
            labels_file.close()
            preds_file.close()

            labels = [l.strip() for l in labels]
            preds = [p.strip() for p in preds]

            f1, precision, recall = conlleval_eval(' '.join(labels), ' '.join(preds))

            results_by_epoch.append((f1, precision, recall,
                                     wandb.run.id,
                                     epoch,
                                     training_args.per_device_train_batch_size,
                                     training_args.learning_rate,
                                     training_args.num_train_epochs,
                                     wandb.config.model_name,
                                     training_args.weight_decay,
                                     training_args.warmup_ratio
                                    ))

        results_by_epoch = pd.DataFrame(results_by_epoch, columns=['f1',
                                                                   'precision',
                                                                   'recall',
                                                                   'run_id',
                                                                   'epoch',
                                                                   'batch_size',
                                                                   'learning_rate',
                                                                   'num_train_epochs',
                                                                   'model_name',
                                                                   'weight_decay',
                                                                   'warmup_ratio'
                                                                   ]
                                        )

        results_by_epoch.to_csv(output_dir+'/dev.metrics')

    elif wandb.config.run_type == 'grid':

        save_plm_metric_table(output_dir, training_args)
        save_plm_metric_table(output_dir, training_args, 'test')

if __name__ == "__main__":
    train()
