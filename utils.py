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
Project module with general definitions and statements.
"""


# standard library imports
import itertools
import pathlib

# third party imports
from Bio import SeqIO


data_directory = pathlib.Path("data")

logging_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"


def fasta_to_dict(fasta_file_path, separator=" "):
    """
    Read a FASTA file to a dictionary with keys the first word of each description
    and values the corresponding sequence.

    Args:
        fasta_file_path (path-like object): FASTA file path
        separator (string): description parts delimiter string
    Returns:
        dict: FASTA entries dictionary mapping the first word of each entry
        description to the corresponding sequence
    """
    fasta_dict = {}

    for fasta_entries in read_fasta_in_chunks(fasta_file_path):
        if fasta_entries[-1] is None:
            fasta_entries = [
                fasta_entry for fasta_entry in fasta_entries if fasta_entry is not None
            ]

        for fasta_entry in fasta_entries:
            description = fasta_entry[0]
            first_word = description.split(separator)[0]
            sequence = fasta_entry[1]

            # verify entry keys are unique
            assert first_word not in fasta_dict, f"{first_word=} already in fasta_dict"
            fasta_dict[first_word] = {"description": description, "sequence": sequence}

    return fasta_dict


def read_fasta_in_chunks(fasta_file_path, num_chunk_entries=1024):
    """
    Read a FASTA file in chunks, returning a list of tuples of two strings,
    the FASTA description line without the leading ">" character, and
    the sequence with any whitespace removed.

    Args:
        fasta_file_path (path-like object): FASTA file path
        num_chunk_entries (int): number of entries in each chunk
    Returns:
        generator that produces lists of FASTA entries
    """
    # Count the number of entries in the FASTA file up to the maximum of
    # the num_chunk_entries chunk size. If the FASTA file has fewer entries
    # than num_chunk_entries, re-assign the latter to that smaller value.
    with open(fasta_file_path) as fasta_file:
        num_entries_counter = 0
        for _ in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
            num_entries_counter += 1
            if num_entries_counter == num_chunk_entries:
                break
        else:
            num_chunk_entries = num_entries_counter

    # read the FASTA file in chunks
    with open(fasta_file_path) as fasta_file:
        fasta_generator = SeqIO.FastaIO.SimpleFastaParser(fasta_file)
        args = [fasta_generator] * num_chunk_entries
        fasta_chunks_iterator = itertools.zip_longest(*args)

        for fasta_entries in fasta_chunks_iterator:
            if fasta_entries[-1] is None:
                fasta_entries = [entry for entry in fasta_entries if entry is not None]
            yield fasta_entries


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024
    return f"{num:.1f} Yi{suffix}"


if __name__ == "__main__":
    print("this is a module file, import to use")
