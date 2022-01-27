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
Pipeline to train a coding vs non-coding ORF classifier using a Transformer architecture.
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

# project imports
from utils import (
    AttributeDict,
    generate_dataloaders,
    log_pytorch_cuda_info,
    logger,
    logging_formatter_time_message,
)


class BinaryClassificationTransformer(pl.LightningModule):
    def __init__(
        self,
        embedding_dimension,
        num_heads,
        depth,
        feedforward_connections,
        sequence_length,
        num_tokens,
        dropout_probability,
    ):
        super().__init__()

        output_size = 1

        self.token_embedding = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embedding_dimension
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=sequence_length, embedding_dim=embedding_dimension
        )

        transformer_blocks = [
            TransformerBlock(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                feedforward_connections=feedforward_connections,
                dropout_probability=dropout_probability,
            )
            for _ in range(depth)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.final_layer = nn.Linear(embedding_dimension, output_size)

    def forward(self, x):
        # generate token embeddings
        token_embeddings = self.token_embedding(x)

        b, t, k = token_embeddings.size()

        # generate position embeddings
        position_embeddings_init = torch.arange(t, device=self.device)
        position_embeddings = self.position_embedding(position_embeddings_init)[None, :, :].expand(b, t, k)

        x = token_embeddings + position_embeddings

        x = self.transformer_blocks(x)

        # average-pool over dimension t
        x = x.mean(dim=1)

        x = self.final_layer(x)

        x = torch.sigmoid(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, embedding_dimension, num_heads):
        super().__init__()

        assert (
            embedding_dimension % num_heads == 0
        ), f"embedding dimension must be divisible by number of heads, got {embedding_dimension=}, {num_heads=}"

        self.num_heads = num_heads

        k = embedding_dimension

        self.to_keys = nn.Linear(k, k * num_heads, bias=False)
        self.to_queries = nn.Linear(k, k * num_heads, bias=False)
        self.to_values = nn.Linear(k, k * num_heads, bias=False)

        self.unify_heads = nn.Linear(num_heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.num_heads

        keys = self.to_keys(x).view(b, t, h, k)
        queries = self.to_queries(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot.shape: (b * h, t, t)

        # scale dot product
        dot = dot / (k ** (1 / 2))

        # get row-wise normalized weights
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unify_heads(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        num_heads,
        feedforward_connections,
        dropout_probability,
    ):
        super().__init__()

        assert (
            feedforward_connections > embedding_dimension
        ), f"feed forward subnet number of connections should be larger than the embedding dimension, got {feedforward_connections=}, {embedding_dimension=}"

        k = embedding_dimension

        self.attention = SelfAttention(k, num_heads=num_heads)

        self.layer_normalization_1 = nn.LayerNorm(k)
        self.layer_normalization_2 = nn.LayerNorm(k)

        self.feed_forward = nn.Sequential(
            nn.Linear(k, feedforward_connections),
            nn.ReLU(),
            nn.Linear(feedforward_connections, k),
            nn.Dropout(dropout_probability),
        )

    def forward(self, x):
        x = self.layer_normalization_1(self.attention(x) + x)
        x = self.layer_normalization_2(self.feed_forward(x) + x)

        return x


class ProteinCodingClassifier(BinaryClassificationTransformer):
    """
    Neural network for protein coding or non-coding classification of DNA sequences.
    """

    def __init__(self, **kwargs):
        self.save_hyperparameters()

        super().__init__(
            embedding_dimension=self.hparams.embedding_dimension,
            num_heads=self.hparams.num_heads,
            depth=self.hparams.transformer_depth,
            feedforward_connections=self.hparams.feedforward_connections,
            sequence_length=self.hparams.sequence_length,
            num_tokens=self.hparams.num_nucleobase_letters,
            dropout_probability=self.hparams.dropout_probability,
        )

        self.dna_sequence_mapper = self.hparams.dna_sequence_mapper

        self.best_validation_accuracy = 0

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
        self.test_precision = torchmetrics.Precision(num_classes=2).to(self.device)
        self.test_recall = torchmetrics.Recall(num_classes=2).to(self.device)

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
        logger.info(
            f"test accuracy: {test_accuracy:.4f} | precision: {precision:.4f} | recall: {recall:.4f}"
        )
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
        threshold = torch.Tensor([threshold_value]).to(device=self.device)

        predictions = (output > threshold).to(dtype=torch.int32)
        return predictions


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

        configuration.feature_encoding = "label"

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

        if args.test:
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

        _, _, test_dataloader = generate_dataloaders(network.hparams)

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
