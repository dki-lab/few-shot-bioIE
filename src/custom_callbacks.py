from transformers import TrainerCallback
import ipdb
import numpy as np
import os
import torch
import wandb

class ReportingCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    """

    def __init__(self, model, trainer, train_dataset, eval_dataset):
        self.model = model
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.evaluate(self.train_dataset, metric_key_prefix='train')

class EvalReportingCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    """

    def __init__(self, model, trainer, train_dataset, eval_dataset, test_dataset, label_list):
        self.model = model
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.label_list = label_list

    def evaluate_model(self,dataset, eval_name, args, state):
        predictions, labels, metrics = self.trainer.predict(dataset, metric_key_prefix=eval_name)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        dataset_file = os.path.join(args.output_dir, "{}.tokens.{}.txt".format(eval_name, int(state.epoch)))
        output_predictions_file = os.path.join(args.output_dir, "{}.predictions.{}.txt".format(eval_name, int(state.epoch)))
        output_labels_file = os.path.join(args.output_dir, "{}.labels.{}.txt".format(eval_name, int(state.epoch)))

        # Save predictions
        with open(dataset_file, "w") as writer:
            for tokens in dataset['tokens']:
                writer.write(" ".join(tokens) + "\n")

        with open(output_predictions_file, "w") as writer:
            for prediction in true_predictions:
                writer.write(" ".join(prediction) + "\n")

        with open(output_labels_file, "w") as writer:
            for label in true_labels:
                writer.write(" ".join(label) + "\n")

        return metrics


    def on_epoch_end(self, args, state, control, **kwargs):

        #Only Evaluate Every 5 Epochs
        if state.epoch % wandb.config.epoch_eval_period == 0:
            train_metrics = self.evaluate_model(self.train_dataset, 'train', args, state)

            # self.trainer.log_metrics("train.metrics.{}".format(state.epoch), train_metrics)
            self.trainer.save_metrics("train.metrics.{}".format(int(state.epoch)), train_metrics)
            wandb.log(train_metrics, step=int(state.epoch))

            eval_metrics = self.evaluate_model(self.eval_dataset, 'dev', args, state)

            # self.trainer.log_metrics("dev.metrics.{}".format(state.epoch), eval_metrics)
            self.trainer.save_metrics("dev.metrics.{}".format(int(state.epoch)), eval_metrics)
            wandb.log(eval_metrics, step=int(state.epoch))

            if args.do_predict:
                test_metrics = self.evaluate_model(self.test_dataset, 'test', args, state)
                self.trainer.save_metrics("test.metrics.{}".format(int(state.epoch)), test_metrics)
                wandb.log(test_metrics, step=int(state.epoch))
