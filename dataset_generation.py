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
Generate a pandas dataframe from raw FASTA files with coding and non-coding sequences
and save it to a pickle file.
"""


# standard library imports
import argparse
import sys

# third party imports
import pandas as pd

from loguru import logger

# project imports
from utils import data_directory, fasta_to_dict, logging_format, sizeof_fmt


def generate_datasets(coding_transcripts_path, non_coding_transcripts_path):
    """
    Generate a pandas dataframe from raw FASTA files with coding and non-coding gene
    transcripts and save it to a pickle file.

    Args:
        coding_transcripts_path (path-like object): path to coding transcripts FASTA file
        non_coding_transcripts_path (path-like object): path to non-coding transcripts
            FASTA file
    """
    logger.info(f"reading FASTA file {coding_transcripts_path} ...")
    coding_transcripts_dict = fasta_to_dict(coding_transcripts_path, separator=";")
    coding_transcripts_list = [
        {
            "transcript_id": transcript_id,
            "description": values_dict["description"],
            "sequence": values_dict["sequence"],
            "coding": True,
        }
        for transcript_id, values_dict in coding_transcripts_dict.items()
    ]
    del coding_transcripts_dict

    logger.info(f"reading FASTA file {non_coding_transcripts_path} ...")
    non_coding_transcripts_dict = fasta_to_dict(
        non_coding_transcripts_path, separator=";"
    )
    non_coding_transcripts_list = [
        {
            "transcript_id": transcript_id,
            "description": values_dict["description"],
            "sequence": values_dict["sequence"],
            "coding": False,
        }
        for transcript_id, values_dict in non_coding_transcripts_dict.items()
    ]
    del non_coding_transcripts_dict

    examples_dictionaries = coding_transcripts_list + non_coding_transcripts_list

    dataframe_columns = ["transcript_id", "description", "sequence", "coding"]
    dataset = pd.DataFrame(examples_dictionaries, columns=dataframe_columns)

    generate_dataset_statistics(dataset)

    # save dataset as a pickle file
    dataset_path = data_directory / "dataset.pickle"
    dataset.to_pickle(dataset_path)
    logger.info(f"dataset saved at {dataset_path}")

    generate_dev_datasets(dataset)


def generate_dev_datasets(dataset, random_seed=7):
    """
    Generate and save subsets of the full dataset for faster loading during development.

    Args:
        dataset (pandas DataFrame): full dataset dataframe
        random_seed (int): random seed to initialize pandas sample random state
    """
    dev_dataset_percentages = [1, 5, 20]

    for dataset_percentage in dev_dataset_percentages:
        dataset_id = f"{dataset_percentage}pct"
        logger.info(f"generating {dataset_percentage}% dev dataset ...")
        fraction = dataset_percentage / 100

        coding = dataset.loc[dataset["coding"] == True]
        non_coding = dataset.loc[dataset["coding"] == False]

        coding = coding.sample(frac=fraction, random_state=random_seed)
        non_coding = non_coding.sample(frac=fraction, random_state=random_seed)

        dev_dataset = pd.concat([coding, non_coding])
        dev_dataset = dev_dataset.sort_index()

        generate_dataset_statistics(dev_dataset)

        # save dataframe to a pickle file
        pickle_path = data_directory / f"{dataset_id}_dataset.pickle"
        dev_dataset.to_pickle(pickle_path)
        logger.info(f"{dataset_percentage}% dev dataset saved at {pickle_path}")


def generate_dataset_statistics(dataset):
    """
    Generate and log dataset statistics.
    """
    num_examples = len(dataset)
    coding_value_counts = dataset["coding"].value_counts()
    num_coding = coding_value_counts[True].item()
    num_non_coding = coding_value_counts[False].item()
    logger.info(
        f"dataset contains {num_coding:,} coding and {num_non_coding:,} non-coding transcripts, {num_examples:,} in total"
    )

    dataset_object_size = sys.getsizeof(dataset)
    logger.info("dataset object memory usage: {}".format(sizeof_fmt(dataset_object_size)))

    dataset["sequence_length"] = dataset["sequence"].str.len()

    sequence_length_mean = dataset["sequence_length"].mean()
    sequence_length_median = dataset["sequence_length"].median()
    sequence_length_standard_deviation = dataset["sequence_length"].std()
    logger.info(
        f"sequences length mean: {sequence_length_mean:.2f}, median: {sequence_length_median:.2f}, standard deviation: {sequence_length_standard_deviation:.2f}"
    )


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--generate_datasets",
        action="store_true",
        help="generate full and dev dataset dataframes saved as pickle files",
    )
    argument_parser.add_argument(
        "--coding_transcripts_path",
        help="coding sequences FASTA path",
    )
    argument_parser.add_argument(
        "--non_coding_transcripts_path",
        help="non-coding sequences FASTA path",
    )

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=logging_format)
    data_directory.mkdir(exist_ok=True)
    log_file_path = data_directory / "dataset_generation.log"
    logger.add(log_file_path, format=logging_format)

    if (
        args.generate_datasets
        and args.coding_transcripts_path
        and args.non_coding_transcripts_path
    ):
        generate_datasets(args.coding_transcripts_path, args.non_coding_transcripts_path)
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
