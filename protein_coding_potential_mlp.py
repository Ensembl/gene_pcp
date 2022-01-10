#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# See the NOTICE file distributed with this work for additional information
# regarding copyright ownership.
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
Train a coding / non-coding ORF classifier.
"""


# standard library imports
import argparse
import datetime as dt
import logging
import pathlib
import random
import sys
import warnings

# third party imports
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import yaml

from torch import nn
from torch.utils.data import DataLoader, random_split

# project imports
from utils import (
    AttributeDict,
    SequenceDataset,
    log_pytorch_cuda_info,
    logger,
    logging_formatter_time_message,
)


class ProteinCodingClassifier(pl.LightningModule):
    """
    Neural network for protein coding or non-coding classification of DNA sequences.
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.sequence_length = self.hparams.sequence_length
        self.padding_side = self.hparams.padding_side
        self.dna_sequence_mapper = self.hparams.dna_sequence_mapper

        input_size = self.sequence_length * self.hparams.num_nucleobase_letters
        output_size = 1

        self.input_layer = nn.Linear(
            in_features=input_size, out_features=self.hparams.num_connections
        )

        # workaround for a bug when saving network to TorchScript format
        self.hparams.dropout_probability = float(self.hparams.dropout_probability)

        self.dropout = nn.Dropout(self.hparams.dropout_probability)
        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(
            in_features=self.hparams.num_connections, out_features=output_size
        )

        self.final_activation = nn.Sigmoid()

        self.best_validation_accuracy = 0

    def forward(self, x):
        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.dropout(x)
        x = self.final_activation(x)

        return x

    def on_pretrain_routine_end(self):
        logger.info("start network training")
        logger.info(f"configuration:\n{self.hparams}")

    def training_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

        labels = labels.unsqueeze(1)
        labels = labels.to(torch.float32)

        training_loss = F.binary_cross_entropy(output, labels)
        self.log("training_loss", training_loss)

        # clip gradients to prevent the exploding gradient problem
        if self.hparams.clip_max_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip_max_norm)

        return training_loss

    def on_validation_start(self):
        self.validation_accuracy = torchmetrics.Accuracy(num_classes=2)

    def validation_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

        labels = labels.unsqueeze(1)
        labels = labels.to(torch.float32)

        validation_loss = F.binary_cross_entropy(output, labels)
        self.log("validation_loss", validation_loss)

        predictions = self.get_predictions(output)

        labels = labels.to(torch.int32)
        self.validation_accuracy(predictions, labels)

    def on_validation_end(self):
        self.best_validation_accuracy = max(
            self.best_validation_accuracy,
            self.validation_accuracy.compute().item(),
        )

    def on_test_start(self):
        # save network in TorchScript format
        experiment_directory_path = pathlib.Path(self.hparams.experiment_directory)
        torchscript_path = experiment_directory_path / "torchscript_network.pt"
        torchscript = self.to_torchscript()
        torch.jit.save(torchscript, torchscript_path)

        self.test_accuracy = torchmetrics.Accuracy(num_classes=2)
        self.test_precision = torchmetrics.Precision(num_classes=2)
        self.test_recall = torchmetrics.Recall(num_classes=2)

    def test_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

        predictions = self.get_predictions(output)

        labels = labels.unsqueeze(1)

        self.test_accuracy(predictions, labels)
        self.test_precision(predictions, labels)
        self.test_recall(predictions, labels)

    def on_test_end(self):
        # log statistics
        test_accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        logger.info(f"test accuracy: {test_accuracy:.4f} | precision: {precision:.4f} | recall: {recall:.4f}")
        logger.info(f"best validation accuracy: {self.best_validation_accuracy:.4f}")

    def configure_optimizers(self):
        # optimization function
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def get_predictions(self, output):
        threshold_value = 0.5
        threshold = torch.Tensor([threshold_value])

        predictions = (output > threshold).int()
        return predictions


def generate_dataloaders(configuration):
    """
    Generate training, validation, and test dataloaders from the dataset files.

    Args:
        configuration (AttributeDict): experiment configuration AttributeDict
    Returns:
        tuple containing the training, validation, and test dataloaders
    """
    dataset = SequenceDataset(
        dataset_id=configuration.dataset_id,
        sequence_length=configuration.sequence_length,
        padding_side=configuration.padding_side,
    )

    configuration.dna_sequence_mapper = dataset.dna_sequence_mapper
    configuration.num_nucleobase_letters = (
        configuration.dna_sequence_mapper.num_nucleobase_letters
    )

    # calculate the training, validation, and test set size
    dataset_size = len(dataset)
    configuration.validation_size = int(configuration.validation_ratio * dataset_size)
    configuration.test_size = int(configuration.test_ratio * dataset_size)
    configuration.training_size = (
        dataset_size - configuration.validation_size - configuration.test_size
    )

    # split dataset into training, validation, and test datasets
    training_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        lengths=(
            configuration.training_size,
            configuration.validation_size,
            configuration.test_size,
        ),
        generator=torch.Generator().manual_seed(configuration.random_seed),
    )

    logger.info(
        f"dataset split to training ({configuration.training_size}), validation ({configuration.validation_size}), and test ({configuration.test_size}) datasets"
    )

    # set the batch size equal to the size of the smallest dataset if larger than that
    configuration.batch_size = min(
        configuration.batch_size,
        configuration.training_size,
        configuration.validation_size,
        configuration.test_size,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=configuration.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=configuration.batch_size,
        num_workers=configuration.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configuration.batch_size,
        num_workers=configuration.num_workers,
    )

    return (training_loader, validation_loader, test_loader)


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--configuration",
        help="path to the experiment configuration file",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")

    args = argument_parser.parse_args()

    # filter warning about number of dataloader workers
    warnings.filterwarnings(
        "ignore",
        ".*does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument.*",
    )

    # train a new classifier
    if args.train and args.configuration:
        # read the experiment configuration YAML file to a dictionary
        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)

        configuration = AttributeDict(configuration)

        configuration.datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")
        configuration.logging_version = f"{configuration.experiment_prefix}_{configuration.dataset_id}_{configuration.datetime}"

        # generate random seed if it doesn't exist
        configuration.random_seed = configuration.get(
            "random_seed", random.randint(1_000_000, 1_001_000)
        )

        configuration.feature_encoding = "one-hot"

        configuration.experiment_directory = f"{configuration.save_directory}/{configuration.logging_version}"
        log_directory_path = pathlib.Path(configuration.experiment_directory)
        log_directory_path.mkdir(parents=True, exist_ok=True)

        # create file handler and add to logger
        log_file_path = log_directory_path / "experiment.log"
        file_handler = logging.FileHandler(log_file_path, mode="a+")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging_formatter_time_message)
        logger.addHandler(file_handler)

        log_pytorch_cuda_info()

        # get training, validation, and test dataloaders
        (
            training_dataloader,
            validation_dataloader,
            test_dataloader,
        ) = generate_dataloaders(configuration)

        # instantiate neural network
        network = ProteinCodingClassifier(**configuration)

        # don't use a per-experiment subdirectory
        logging_name = ""

        tensorboard_logger = pl.loggers.TensorBoardLogger(
            save_dir=configuration.save_directory,
            name=logging_name,
            version=configuration.logging_version,
            default_hp_metric=False,
        )

        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="validation_loss",
            min_delta=configuration.loss_delta,
            patience=configuration.patience,
            verbose=True,
        )

        trainer = pl.Trainer(
            logger=tensorboard_logger,
            max_epochs=configuration.max_epochs,
            log_every_n_steps=1,
            callbacks=[early_stopping_callback],
            profiler=configuration.profiler,
        )

        trainer.fit(
            model=network,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
        )

        if args.test:
            trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    else:
        argument_parser.print_help()
        sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
