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
Pipeline to train a coding vs non-coding ORF classifier using a multilayer perceptron architecture.
"""


# standard library imports
import argparse
import datetime as dt
import logging
import math
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

# project imports
from utils import (
    AttributeDict,
    generate_dataloaders,
    log_pytorch_cuda_info,
    logger,
    logging_formatter_time_message,
    prettify_confusion_matrix,
)


class ProteinCodingClassifier(pl.LightningModule):
    """
    Neural network for protein coding or non-coding classification of DNA sequences.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.dna_sequence_mapper = self.hparams.dna_sequence_mapper

        # the number of residues is "biologically hardcoded"
        num_residues = 20
        input_size = int(
            math.factorial(num_residues)
            / math.factorial(num_residues - self.hparams.window_length)
        )
        output_size = 1

        self.num_connections = self.hparams.num_connections

        self.input_layer = nn.Linear(
            in_features=input_size, out_features=self.num_connections
        )

        self.dropout = nn.Dropout(self.hparams.dropout_probability)
        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(
            in_features=self.num_connections, out_features=output_size
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

        # loss function
        training_loss = F.binary_cross_entropy(output, labels)
        self.log("training_loss", training_loss)

        # clip gradients to prevent the exploding gradient problem
        if self.hparams.clip_max_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip_max_norm)

        return training_loss

    def on_validation_start(self):
        # https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-and-devices
        self.validation_accuracy = torchmetrics.Accuracy(num_classes=2).to(self.device)

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

    def on_train_end(self):
        # NOTE: disabling saving network to TorchScript, seems buggy

        # workaround for a bug when saving network to TorchScript format
        # self.hparams.dropout_probability = float(self.hparams.dropout_probability)

        # save network to TorchScript format
        # experiment_directory_path = pathlib.Path(self.hparams.experiment_directory)
        # torchscript_path = experiment_directory_path / "torchscript_network.pt"
        # torchscript = self.to_torchscript()
        # torch.jit.save(torchscript, torchscript_path)
        pass

    def on_test_start(self):
        self.test_accuracy = torchmetrics.Accuracy(num_classes=2).to(self.device)
        self.test_precision = torchmetrics.Precision(num_classes=2, average=None).to(
            self.device
        )
        self.test_recall = torchmetrics.Recall(num_classes=2, average=None).to(
            self.device
        )
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2).to(
            self.device
        )
        self.test_auroc = torchmetrics.AUROC(num_classes=1).to(self.device)

    def test_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

        predictions = self.get_predictions(output)

        labels = labels.unsqueeze(1)

        self.test_accuracy(predictions, labels)
        self.test_precision(predictions, labels)
        self.test_recall(predictions, labels)
        self.test_confusion_matrix(predictions, labels)
        self.test_auroc(predictions, labels)

    def on_test_end(self):
        # log statistics
        test_accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        confusion_matrix = self.test_confusion_matrix.compute()
        auroc = self.test_auroc.compute()

        labels = ["non-coding", "coding"]
        confusion_matrix_string = prettify_confusion_matrix(
            confusion_matrix, labels, reverse_order=True
        )

        logger.info(
            f"test accuracy: {test_accuracy:.4f} (best validation accuracy: {self.best_validation_accuracy:.4f})"
        )
        logger.info(f"precision: {precision[1]:.4f} | recall: {recall[1]:.4f}")
        logger.info(f"confusion matrix:\n{confusion_matrix_string}")
        logger.info(f"AUROC: {auroc:.4f}")

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
        threshold = torch.Tensor([threshold_value]).to(device=self.device)

        predictions = (output > threshold).to(dtype=torch.int32)
        return predictions


def get_item_freq_features(self, index):
    """
    Modularized Dataset __getitem__ method.

    Generate a feature vector from the frequencies of all permutations of aminoacids.

    Args:
        self (Dataset): the Dataset object that will contain __getitem__
    Returns:
        tuple containing the features vector and sequence coding value
    """
    sample = self.dataset.iloc[index].to_dict()

    sequence = sample["sequence"]
    coding = sample["coding"]

    coding_value = int(coding)

    freq_sequence = self.dna_sequence_mapper.sequence_to_freq(sequence)

    item = (freq_sequence, coding_value)

    return item


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--datetime",
        help="datetime string; if set this will be used instead of generating a new one",
    )
    argument_parser.add_argument(
        "--configuration",
        help="path to the experiment configuration file",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")
    argument_parser.add_argument("--checkpoint", help="experiment checkpoint path")

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

        if args.datetime:
            configuration.datetime = args.datetime
        else:
            configuration.datetime = dt.datetime.now().isoformat(
                sep="_", timespec="seconds"
            )

        configuration.logging_version = f"{configuration.experiment_prefix}_{configuration.dataset_id}_{configuration.datetime}"

        # generate random seed if it doesn't exist
        # Using the range [1_000_000, 1_001_000] for the random seed. This range contains
        # numbers that have a good balance of 0 and 1 bits, as recommended by the PyTorch docs.
        # https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.manual_seed
        configuration.random_seed = configuration.get(
            "random_seed", random.randint(1_000_000, 1_001_000)
        )

        configuration.experiment_directory = (
            f"{configuration.save_directory}/{configuration.logging_version}"
        )
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
        ) = generate_dataloaders(configuration, get_item_freq_features)

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
            gpus=configuration.gpus,
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

        trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    # test a trained classifier
    elif args.test and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        logging_directory = checkpoint_path.with_suffix("")
        logging_directory.mkdir(exist_ok=True)

        # create file handler and add to logger
        log_file_path = logging_directory / f"{checkpoint_path.stem}.log"

        file_handler = logging.FileHandler(log_file_path, mode="a+")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging_formatter_time_message)
        logger.addHandler(file_handler)

        network = ProteinCodingClassifier.load_from_checkpoint(checkpoint_path)

        _, _, test_dataloader = generate_dataloaders(
            network.hparams, get_item_freq_features
        )

        tensorboard_logger = pl.loggers.TensorBoardLogger(
            save_dir=logging_directory,
            default_hp_metric=False,
        )

        trainer = pl.Trainer(logger=tensorboard_logger)
        trainer.test(network, dataloaders=test_dataloader)

    else:
        argument_parser.print_help()
        sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
