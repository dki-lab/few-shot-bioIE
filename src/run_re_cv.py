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

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AdamW
from transformers.utils.versions import require_version
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset

import ipdb
import torch
import random
import wandb
from glob import glob
from eval_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

pos_labels_by_dataset = {'DDI':['DDI-mechanism', 'DDI-advise', 'DDI-effect','DDI-int'],
                         'chemprot':['CPR:4', 'CPR:6', 'CPR:5', 'CPR:9', 'CPR:3'],
                         'gad':['1']}

def train_epoch(model, train_dataloader, optimizer, scheduler):
    print("")
    print('Training...')

    model.train()
    total_train_loss = 0

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss, logits = outputs['loss'], outputs['logits']
        wandb.log({'train_batch_loss': loss.item()})
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    wandb.log({'avg_train_loss': avg_train_loss})

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

def evaluate(model, dev_dataloader, dev_df, labels_to_id, eval_name, args, epoch):

    if epoch % wandb.config.epoch_eval_period == 0:
        print("")
        print("Running Validation...")

        id_to_labels = {i:str(l) for l,i in labels_to_id.items()}
        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0

        predictions_list = []
        labels_list = []
        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss, logits = outputs['loss'], outputs['logits']
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            #             total_eval_accuracy = flatten(logits, label_ids)
            predictions_list.extend(pred_flat)
            labels_list.extend(labels_flat)

        dataset_file = os.path.join(args.output_dir, "{}.tokens.{}.txt".format(eval_name, int(epoch)))
        output_predictions_file = os.path.join(args.output_dir, "{}.predictions.{}.txt".format(eval_name, int(epoch)))
        output_labels_file = os.path.join(args.output_dir, "{}.labels.{}.txt".format(eval_name, int(epoch)))

        # Save predictions
        with open(dataset_file, "w") as writer:
            for sent in dev_df['masked_sents']:
                writer.write(sent + "\n")

        with open(output_predictions_file, "w") as writer:
            for prediction in predictions_list:
                writer.write(str(id_to_labels[prediction]) + "\n")

        with open(output_labels_file, "w") as writer:
            for label in labels_list:
                writer.write(str(id_to_labels[label]) + "\n")

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dev_dataloader)
        wandb.log({'val_accuracy': avg_val_accuracy, 'avg_val_loss': avg_val_loss})

        pos_labels = pos_labels_by_dataset[wandb.config.data_name]
        f1 = precision_recall_fscore_support(labels_list, predictions_list, average=None)
        p, r, f, s = precision_recall_fscore_support(y_pred=predictions_list, y_true=labels_list, labels=[labels_to_id[l] for l in pos_labels],
                                                     average='micro')

        metrics = {'{}_f1'.format(eval_name):f,
                   '{}_precision'.format(eval_name):p,
                   '{}_recall'.format(eval_name):r,
                   '{}_class_f1s'.format(eval_name): list(f1[2])}
        json.dump(metrics, open(args.output_dir + '/{}.metrics.{}_results.json'.format(eval_name,epoch),'w'))

hyperparameter_defaults = dict(
    data_name='DDI',
    data_dir='../data/',
    output_dir='../outputs/',
    subset_num=1,
    experiment_num=1,
    run_type='grid',#cv, grid, best, worst
    kfolds=1,
    do_train=True,
    do_eval=True,
    do_predict=False,

    per_device_train_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.1,
    warmup_ratio=0.06,

    model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    per_device_eval_batch_size=16,
    seed=42,
    logging_strategy='no',
    evaluation_strategy='steps',
    save_strategy='no',
    eval_steps_epochs=None,
    save_total_limit=0,
    max_seq_length=128,
    max_eval_samples=None,
    max_predict_samples=None,
    overwrite_output_dir=True,
    gradient_checkpointing=True,
    disable_tqdm=True,
    epoch_eval_period=1
)

wandb.init(config=hyperparameter_defaults, project="bertNER")
config = wandb.config

def train():
    dataset_dir = os.path.join(wandb.config.data_dir, wandb.config.data_name)

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

    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_name, do_lower_cased=True)
    print(len(tokenizer))
    special_tokens_dict = {'additional_special_tokens': ['ENT1', 'ENT2']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    train_df, dev_df, test_df = load_re_dataset(dataset_dir, wandb.config.subset_num)

    unique_labels = np.sort(train_df['label'].unique(),kind='stable')
    labels_to_id = {str(l):i for l,i in zip(unique_labels, range(len(unique_labels)))}
    num_labels = len(unique_labels)

    #Loading Test Dataset before k-fold CV
    if training_args.do_predict:
        test_dataset = get_re_dataset(test_df, tokenizer, labels_to_id)

        test_dataloader = DataLoader(
            test_dataset,  # The validation samples.
            sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
            batch_size=wandb.config.per_device_eval_batch_size  # Evaluates with this batch size.
        )
    else:
        test_dataloader = None

    original_train_dataset = train_df.sample(len(train_df), random_state=np.random.RandomState(wandb.config.seed)).reset_index()
    fold_size = int(len(original_train_dataset) / wandb.config.kfolds)

    for fold in range(wandb.config.kfolds):

        if wandb.config.kfolds > 1:
            training_args.output_dir = os.path.join(output_dir, str(fold))
            os.makedirs(training_args.output_dir)


            train_df = original_train_dataset[(original_train_dataset.index < fold * fold_size) | (original_train_dataset.index >= (fold + 1) * fold_size)]
            dev_df = original_train_dataset[(original_train_dataset.index >= fold * fold_size) & (original_train_dataset.index < (fold + 1) * fold_size)]

            assert len(dev_df) == fold_size
            assert len(train_df) == len(original_train_dataset) - fold_size

        if wandb_dict['eval_steps_epochs'] is not None:
            training_args.eval_steps = wandb_dict['eval_steps_epochs'] * (
                        int(len(original_train_dataset) / training_args.per_device_train_batch_size) + 1)

        train_dataset = get_re_dataset(train_df, tokenizer, labels_to_id)
        dev_dataset = get_re_dataset(dev_df, tokenizer, labels_to_id)

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=wandb.config.per_device_train_batch_size  # Trains with this batch size.
        )

        dev_dataloader = DataLoader(
            dev_dataset,  # The validation samples.
            sampler=SequentialSampler(dev_dataset),  # Pull out batches sequentially.
            batch_size=wandb.config.per_device_eval_batch_size  # Evaluates with this batch size.
        )

        model = AutoModelForSequenceClassification.from_pretrained(wandb.config.model_name,
                                                                   num_labels=num_labels,
                                                                   output_attentions=False,
                                                                   # Whether the model returns attentions weights.
                                                                   output_hidden_states=False,
                                                                   # Whether the model returns all hidden-states.
                                                                   )
        model.resize_token_embeddings(len(tokenizer))
        model.gradient_checkpointing_enable()
        model.to(device)

        print('Learning_rate = ', wandb.config.learning_rate)
        optimizer = AdamW(model.parameters(),
                          lr=wandb.config.learning_rate,
                          weight_decay=wandb.config.weight_decay
                          )

        epochs = wandb.config.num_train_epochs
        warmup_ratio = wandb.config.warmup_ratio
        #     ipdb.set_trace()
        print('epochs =>', epochs)
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        num_warmup_steps = warmup_ratio * total_steps
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        # Training
        if training_args.do_train:

            for epoch_i in range(1, epochs+1):
                print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))

                train_epoch(model, train_dataloader, optimizer, scheduler)

                evaluate(model, dev_dataloader, dev_df, labels_to_id, 'dev', training_args, epoch_i)
                if training_args.do_predict:
                    evaluate(model, test_dataloader, test_df, labels_to_id, 'test', training_args, epoch_i)

        # wandb.log({'training_data_size': len(train_dataset)})

        # if training_args.save_strategy != 'no':
        model.save_pretrained(output_dir + '/trained_model_fold_{}'.format(fold))

        del model
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

            preds = [labels_to_id[p.strip()] for p in preds]
            labels = [labels_to_id[l.strip()] for l in labels]

            pos_labels = pos_labels_by_dataset[wandb.config.data_name]
            p, r, f, s = precision_recall_fscore_support(y_pred=preds, y_true=labels,
                                                         labels=[labels_to_id[l] for l in pos_labels],
                                                         average='micro')

            results_by_epoch.append((f, p, r,
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
