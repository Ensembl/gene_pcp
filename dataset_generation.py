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
from utils import data_directory, fasta_to_dict, logging_format


def generate_dataset_pickle(coding_transcripts_path, non_coding_transcripts_path):
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

    # save dataset as a pickle file
    dataset_path = data_directory / "dataset.pickle"
    dataset.to_pickle(dataset_path)
    logger.info(f"dataset saved at {dataset_path}")


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--generate_dataset_pickle",
        action="store_true",
        help="generate pickled pandas dataframe with all dataset samples",
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
        args.generate_dataset_pickle
        and args.coding_transcripts_path
        and args.non_coding_transcripts_path
    ):
        generate_dataset_pickle(
            args.coding_transcripts_path, args.non_coding_transcripts_path
        )
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
