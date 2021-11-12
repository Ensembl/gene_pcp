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
import math
import pathlib
import pprint
import random
import sys
import time

# third party imports
import numpy as np
import torch
import torchmetrics
import yaml

from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# project imports
from utils import SequenceDataset, data_directory, logging_format


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProteinCodingClassifier(nn.Module):
    """
    A neural network for classification of DNA sequences to protein coding or non-coding.
    """

    def __init__(
        self,
        sequence_length,
        padding_side,
        num_nucleobase_letters,
        num_connections,
        dropout_probability,
        dna_sequence_mapper,
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.padding_side = padding_side
        self.dropout_probability = dropout_probability
        self.dna_sequence_mapper = dna_sequence_mapper

        input_size = self.sequence_length * num_nucleobase_letters
        output_size = 1

        self.input_layer = nn.Linear(in_features=input_size, out_features=num_connections)
        if self.dropout_probability > 0:
            self.dropout = nn.Dropout(self.dropout_probability)

        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(
            in_features=num_connections, out_features=output_size
        )

        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Perform a forward pass of the network.
        """
        x = self.input_layer(x)
        if self.dropout_probability > 0:
            x = self.dropout(x)
        x = self.relu(x)

        x = self.output_layer(x)
        if self.dropout_probability > 0:
            x = self.dropout(x)
        x = self.final_activation(x)

        return x

    # @staticmethod
    def get_predictions(self, output):
        """
        Get predictions.
        """
        threshold_value = 0.5
        threshold = torch.Tensor([threshold_value])

        predictions = (output > threshold).int()
        return predictions


class EarlyStopping:
    """
    Stop training if validation loss doesn't improve during a specified patience period.
    """

    def __init__(self, patience=7, loss_delta=0):
        """
        Args:
            checkpoint_path (path-like object): Path to save the checkpoint.
            patience (int): Number of calls to continue training if validation loss is not improving. Defaults to 7.
            loss_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.loss_delta = loss_delta

        self.no_progress = 0
        self.min_validation_loss = np.Inf

    def __call__(
        self,
        network,
        optimizer,
        experiment,
        validation_loss,
        checkpoint_path,
    ):
        if self.min_validation_loss == np.Inf:
            self.min_validation_loss = validation_loss
            logger.info("saving first network checkpoint...")
            checkpoint = {
                "experiment": experiment,
                "network_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

        elif validation_loss <= self.min_validation_loss - self.loss_delta:
            validation_loss_decrease = self.min_validation_loss - validation_loss
            assert (
                validation_loss_decrease > 0
            ), f"{validation_loss_decrease=}, should be a positive number"
            logger.info(
                f"validation loss decreased by {validation_loss_decrease:.4f}, saving network checkpoint..."
            )

            self.min_validation_loss = validation_loss
            self.no_progress = 0
            checkpoint = {
                "experiment": experiment,
                "network_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

        else:
            self.no_progress += 1

            if self.no_progress == self.patience:
                logger.info(
                    f"{self.no_progress} epochs with no validation loss improvement, stopping training"
                )
                return True

        return False


class Experiment:
    """
    Object containing settings values and status of an experiment.
    """

    def __init__(self, experiment_settings, datetime):
        for attribute, value in experiment_settings.items():
            setattr(self, attribute, value)

        # experiment parameters
        self.datetime = datetime

        # set a seed for the PyTorch random number generator if not present
        if not hasattr(self, "random_seed"):
            self.random_seed = random.randint(1, 100)

        # early stopping
        loss_delta = 0.001
        self.stop_early = EarlyStopping(self.patience, loss_delta)

        # loss function
        self.criterion = nn.BCELoss()

        self.num_complete_epochs = 0

        self.filename = f"{self.filename_prefix}_{self.dataset_id}_{self.datetime}"

        # self.padding_side = "left"
        self.padding_side = "right"

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


def generate_dataloaders(experiment):
    """
    Generate training, validation, and test dataloaders from the dataset files.

    Args:
        experiment (Experiment): Experiment object containing metadata
    Returns:
        tuple containing the training, validation, and test dataloaders
    """
    dataset = SequenceDataset(
        dataset_id=experiment.dataset_id,
        sequence_length=experiment.sequence_length,
        padding_side=experiment.padding_side,
    )

    experiment.dna_sequence_mapper = dataset.dna_sequence_mapper
    experiment.num_nucleobase_letters = (
        experiment.dna_sequence_mapper.num_nucleobase_letters
    )

    # calculate the training, validation, and test set size
    dataset_size = len(dataset)
    experiment.validation_size = int(experiment.validation_ratio * dataset_size)
    experiment.test_size = int(experiment.test_ratio * dataset_size)
    experiment.training_size = (
        dataset_size - experiment.validation_size - experiment.test_size
    )

    # split dataset into training, validation, and test datasets
    training_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        lengths=(
            experiment.training_size,
            experiment.validation_size,
            experiment.test_size,
        ),
    )

    logger.info(
        f"dataset split to training ({experiment.training_size}), validation ({experiment.validation_size}), and test ({experiment.test_size}) datasets"
    )

    # set the batch size equal to the size of the smallest dataset if larger than that
    experiment.batch_size = min(
        experiment.batch_size,
        experiment.training_size,
        experiment.validation_size,
        experiment.test_size,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        num_workers=experiment.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        num_workers=experiment.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        num_workers=experiment.num_workers,
    )

    return (training_loader, validation_loader, test_loader)


def load_checkpoint(checkpoint_path):
    """
    Load an experiment checkpoint and return the experiment, network, and optimizer objects.


    Args:
        checkpoint_path (path-like object): path to the saved experiment checkpoint
    Returns:
        tuple[Experiment, torch.nn.Module, torch.optim.Optimizer] containing
        the experiment state, classifier network, and optimizer
    """
    logger.info(f'loading experiment checkpoint "{checkpoint_path}" ...')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    logger.info(f'"{checkpoint_path}" experiment checkpoint loaded')

    experiment = checkpoint["experiment"]

    network = ProteinCodingClassifier(
        experiment.sequence_length,
        experiment.padding_side,
        experiment.num_nucleobase_letters,
        experiment.num_connections,
        experiment.dropout_probability,
        experiment.dna_sequence_mapper,
    )
    network.load_state_dict(checkpoint["network_state_dict"])
    network.to(DEVICE)

    optimizer = torch.optim.Adam(network.parameters(), lr=experiment.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (experiment, network, optimizer)


def train_network(
    network,
    optimizer,
    experiment,
    training_loader,
    validation_loader,
):
    tensorboard_log_dir = f"runs/{experiment.dataset_id}/{experiment.datetime}"
    summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    max_epochs = experiment.max_epochs
    criterion = experiment.criterion

    checkpoint_path = f"{experiment.experiment_directory}/{experiment.filename}.pth"
    logger.info(f"start training, experiment checkpoints saved at {checkpoint_path}")

    path = pathlib.Path(checkpoint_path)
    network_path = pathlib.Path(f"{path.parent}/{path.stem}_network.pth")
    torch.save(network, network_path)
    logger.info(f"initial raw network saved at {network_path}")

    max_epochs_length = len(str(max_epochs))

    num_train_batches = math.ceil(experiment.training_size / experiment.batch_size)
    num_batches_length = len(str(num_train_batches))

    if not hasattr(experiment, "average_training_losses"):
        experiment.average_training_losses = []

    if not hasattr(experiment, "average_validation_losses"):
        experiment.average_validation_losses = []

    experiment.epoch = experiment.num_complete_epochs + 1
    epoch_times = []
    for epoch in range(experiment.epoch, max_epochs + 1):
        epoch_start_time = time.time()

        experiment.epoch = epoch

        # training
        ########################################################################
        training_losses = []
        # https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices
        train_accuracy = torchmetrics.Accuracy(num_classes=2).to(DEVICE)

        # set the network in training mode
        network.train()

        batch_execution_times = []
        batch_loading_times = []
        pre_batch_loading_time = time.time()
        for batch_number, (inputs, labels) in enumerate(training_loader, start=1):
            batch_start_time = time.time()
            batch_loading_time = batch_start_time - pre_batch_loading_time
            if batch_number < num_train_batches:
                batch_loading_times.append(batch_loading_time)

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            labels = labels.unsqueeze(1)
            labels = labels.to(torch.float32)

            # zero accumulated gradients
            optimizer.zero_grad()

            # forward pass
            output = network(inputs)

            # get predictions
            predictions = network.get_predictions(output)

            # compute training loss
            training_loss = criterion(output, labels)
            training_losses.append(training_loss.item())
            summary_writer.add_scalar("loss/training", training_loss, epoch)

            # perform back propagation
            training_loss.backward()

            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(network.parameters(), experiment.clip_max_norm)

            # perform an optimization step
            optimizer.step()

            labels = labels.to(torch.int32)
            batch_train_accuracy = train_accuracy(predictions, labels)
            average_training_loss = np.average(training_losses)

            batch_finish_time = time.time()
            pre_batch_loading_time = batch_finish_time
            batch_execution_time = batch_finish_time - batch_start_time
            if batch_number < num_train_batches:
                batch_execution_times.append(batch_execution_time)

            train_progress = f"epoch {epoch:{max_epochs_length}} batch {batch_number:{num_batches_length}} of {num_train_batches} | average loss: {average_training_loss:.4f} | accuracy: {batch_train_accuracy:.4f} | execution: {batch_execution_time:.2f}s | loading: {batch_loading_time:.2f}s"
            logger.info(train_progress)

        experiment.num_complete_epochs += 1

        average_training_loss = np.average(training_losses)
        experiment.average_training_losses.append(average_training_loss)

        # validation
        ########################################################################
        num_validation_batches = math.ceil(
            experiment.validation_size / experiment.batch_size
        )
        num_batches_length = len(str(num_validation_batches))

        validation_losses = []
        validation_accuracy = torchmetrics.Accuracy(num_classes=2).to(DEVICE)

        # disable gradient calculation
        with torch.no_grad():
            # set the network in evaluation mode
            network.eval()
            for batch_number, (inputs, labels) in enumerate(validation_loader, start=1):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                labels = labels.unsqueeze(1)
                labels = labels.to(torch.float32)

                # forward pass
                output = network(inputs)

                # get predictions
                predictions = network.get_predictions(output)

                # compute validation loss
                validation_loss = criterion(output, labels)
                validation_losses.append(validation_loss.item())
                summary_writer.add_scalar("loss/validation", validation_loss, epoch)

                labels = labels.to(torch.int32)
                batch_validation_accuracy = validation_accuracy(predictions, labels)
                average_validation_loss = np.average(validation_losses)

                validation_progress = f"epoch {epoch:{max_epochs_length}} validation batch {batch_number:{num_batches_length}} of {num_validation_batches} | average loss: {average_validation_loss:.4f} | accuracy: {batch_validation_accuracy:.4f}"
                logger.info(validation_progress)

        average_validation_loss = np.average(validation_losses)
        experiment.average_validation_losses.append(average_validation_loss)

        total_validation_accuracy = validation_accuracy.compute()

        average_batch_execution_time = sum(batch_execution_times) / len(
            batch_execution_times
        )
        average_batch_loading_time = sum(batch_loading_times) / len(batch_loading_times)

        epoch_finish_time = time.time()
        epoch_time = epoch_finish_time - epoch_start_time
        epoch_times.append(epoch_time)

        train_progress = f"epoch {epoch:{max_epochs_length}} complete | validation loss: {average_validation_loss:.4f} | validation accuracy: {total_validation_accuracy:.4f} | time: {epoch_time:.2f}s"
        logger.info(train_progress)
        logger.info(
            f"training batch average execution time: {average_batch_execution_time:.2f}s | average loading time: {average_batch_loading_time:.2f}s ({num_train_batches - 1} complete batches)"
        )

        if experiment.stop_early(
            network,
            optimizer,
            experiment,
            average_validation_loss,
            checkpoint_path,
        ):
            summary_writer.flush()
            summary_writer.close()
            break

    training_time = sum(epoch_times)
    average_epoch_time = training_time / len(epoch_times)
    logger.info(
        f"total training time: {training_time:.2f}s | epoch average training time: {average_epoch_time:.2f}s ({epoch} epochs)"
    )

    return checkpoint_path


def test_network(checkpoint_path):
    """
    Calculate test loss and generate metrics.
    """
    experiment, network, _optimizer = load_checkpoint(checkpoint_path)

    logger.info("start testing classifier")
    logger.info(f"experiment:\n{experiment}")
    logger.info(f"network:\n{network}")

    # get test dataloader
    _, _, test_loader = generate_dataloaders(experiment)

    criterion = experiment.criterion

    num_test_batches = math.ceil(experiment.test_size / experiment.batch_size)
    num_batches_length = len(str(num_test_batches))

    test_losses = []
    test_accuracy = torchmetrics.Accuracy(num_classes=2).to(DEVICE)
    test_precision = torchmetrics.Precision(num_classes=2).to(DEVICE)
    test_recall = torchmetrics.Recall(num_classes=2).to(DEVICE)

    with torch.no_grad():
        network.eval()

        for batch_number, (inputs, labels) in enumerate(test_loader, start=1):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            labels = labels.unsqueeze(1)
            labels = labels.to(torch.float32)

            # forward pass
            output = network(inputs)

            # get predictions
            predictions = network.get_predictions(output)

            # calculate test loss
            test_loss = criterion(output, labels)
            test_losses.append(test_loss.item())

            labels = labels.to(torch.int32)
            batch_accuracy = test_accuracy(predictions, labels)
            test_precision(predictions, labels)
            test_recall(predictions, labels)

            logger.info(
                f"test batch {batch_number:{num_batches_length}} of {num_test_batches} | accuracy: {batch_accuracy:.4f}"
            )

    # log statistics
    average_test_loss = np.mean(test_losses)
    total_test_accuracy = test_accuracy.compute()
    precision = test_precision.compute()
    recall = test_recall.compute()
    logger.info(
        f"testing complete | average loss: {average_test_loss:.4f} | accuracy: {total_test_accuracy:.4f}"
    )
    logger.info(f"precision: {precision:.4f} | recall: {recall:.4f}")


def log_pytorch_cuda_info():
    """
    Log PyTorch and CUDA info and device to be used.
    """
    logger.debug(f"{torch.__version__=}")
    logger.debug(f"{DEVICE=}")
    logger.debug(f"{torch.version.cuda=}")
    logger.debug(f"{torch.backends.cudnn.enabled=}")
    logger.debug(f"{torch.cuda.is_available()=}")

    if torch.cuda.is_available():
        logger.debug(f"{torch.cuda.device_count()=}")
        logger.debug(f"{torch.cuda.get_device_properties(DEVICE)}")


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--datetime",
        help="datetime string; if not set it will be generated from the current datetime",
    )
    argument_parser.add_argument(
        "-ex",
        "--experiment_settings",
        help="path to the experiment settings configuration file",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=logging_format)

    # train a new classifier
    if args.train and args.experiment_settings:
        # read the experiment settings YAML file to a dictionary
        with open(args.experiment_settings) as file:
            experiment_settings = yaml.safe_load(file)

        if args.datetime is None:
            datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")
        else:
            datetime = args.datetime

        # generate new experiment
        experiment = Experiment(experiment_settings, datetime)

        pathlib.Path(experiment.experiment_directory).mkdir(exist_ok=True)
        log_file_path = f"{experiment.experiment_directory}/{experiment.filename}.log"
        logger.add(log_file_path, format=logging_format)

        log_pytorch_cuda_info()

        torch.manual_seed(experiment.random_seed)

        # get training, validation, and test dataloaders
        training_loader, validation_loader, _test_loader = generate_dataloaders(
            experiment
        )

        # instantiate neural network
        network = ProteinCodingClassifier(
            experiment.sequence_length,
            experiment.padding_side,
            experiment.num_nucleobase_letters,
            experiment.num_connections,
            experiment.dropout_probability,
            experiment.dna_sequence_mapper,
        )
        network.to(DEVICE)

        # optimization function
        optimizer = torch.optim.Adam(network.parameters(), lr=experiment.learning_rate)

        logger.info("start training new classifier")
        logger.info(f"experiment:\n{experiment}")
        logger.info(f"network:\n{network}")

        checkpoint_path = train_network(
            network,
            optimizer,
            experiment,
            training_loader,
            validation_loader,
        )

        if args.test:
            test_network(checkpoint_path)

    else:
        argument_parser.print_help()
        sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
