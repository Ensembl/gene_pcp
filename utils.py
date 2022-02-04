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
import logging
import pathlib
import sys

# third party imports
import pandas as pd
import torch
import torch.nn.functional as F

from Bio import SeqIO
from Bio.Seq import Seq
from torch.utils.data import DataLoader, Dataset, random_split


data_directory = pathlib.Path("data")

# logging formats
logging_formatter_time_message = logging.Formatter(
    fmt="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging_formatter_message = logging.Formatter(fmt="%(message)s")

# set up base logger
logger = logging.getLogger("main_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False
# create console handler and add to logger
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging_formatter_time_message)
logger.addHandler(console_handler)


class DnaSequenceMapper:
    """
    DNA sequences translation to one-hot or label encoding.
    """

    def __init__(self):
        nucleobase_symbols = ["A", "C", "G", "T", "N"]
        padding_character = [" "]

        self.nucleobase_letters = sorted(nucleobase_symbols + padding_character)

        self.num_nucleobase_letters = len(self.nucleobase_letters)

        self.nucleobase_letter_to_index = {
            nucleobase_letter: index
            for index, nucleobase_letter in enumerate(self.nucleobase_letters)
        }

        self.index_to_nucleobase_letter = {
            index: nucleobase_letter
            for index, nucleobase_letter in enumerate(self.nucleobase_letters)
        }

    def sequence_to_one_hot(self, sequence):
        residues_symbols = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        sequence_indexes =[]

        for i in sequence:
            sequence_translated_list = [sequence[i].split()]
            frec = {}
            n = 0

            while sequence_translated_list:
                triplet = sequence_translated_list[0] + sequence_translated_list[1] + sequence_translated_list[2]
                if triplet in frec:
                    frec[triplet] += 1
                else:
                    frec[triplet] = 1
                n += 1
                del sequence_translated_list[0]
        
            sequence_frecs = []
            for combination in list(combinations(residues_symbols, 3)):
                if frec[combination is not None:
                    sequence_frecs.append(frec[combination]/n)
                else:
                    sequence_frecs.append('0')

        sequence_indexes.append(sequence_frecs)
        
        one_hot_sequence = F.one_hot(
            torch.tensor(sequence_indexes), num_classes=8000
            )
        one_hot_sequence = one_hot_sequence.type(torch.float32)

        return one_hot_sequence

    def sequence_to_label_encoding(self, sequence):
        label_encoded_sequence = [
            self.nucleobase_letter_to_index[nucleobase] for nucleobase in sequence
        ]

        label_encoded_sequence = torch.tensor(label_encoded_sequence, dtype=torch.int32)

        return label_encoded_sequence


class DnaSequenceDataset(Dataset):
    """
    DNA sequences Dataset.
    """

    def __init__(
        self, dataset_id, sequence_length, feature_encoding, padding_side="right"
    ):
        self.dataset_id = dataset_id
        self.sequence_length = sequence_length
        self.feature_encoding = feature_encoding
        self.padding_side = padding_side

        if self.dataset_id == "full":
            dataset_path = data_directory / "dataset.pickle"
            logger.info(f"loading full dataset {dataset_path} ...")
            dataset = pd.read_pickle(dataset_path)
            logger.info("full dataset loaded")
        elif self.dataset_id in ["1pct", "5pct", "20pct"]:
            dev_dataset_path = data_directory / f"{self.dataset_id}_dataset.pickle"
            logger.info(f"loading {self.dataset_id} dev dataset...")
            dataset = pd.read_pickle(dev_dataset_path)
            logger.info(f"{self.dataset_id} dev dataset loaded")

        # select the features and labels columns
        self.dataset = dataset[["sequence", "coding"]]

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            self.dataset["sequence"] = self.dataset["sequence"].str.pad(
                width=sequence_length, side=padding_side, fillchar=" "
            )
            self.dataset["sequence"] = self.dataset["sequence"].str.slice(
                stop=sequence_length
            )

        # generate DNA sequences mapper
        self.dna_sequence_mapper = DnaSequenceMapper()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.iloc[index].to_dict()

        sequence = Seq(sample["sequence"])
        coding = sample["coding"]

        coding_value = int(coding)

        if self.feature_encoding == "one-hot":
            one_hot_sequence = self.dna_sequence_mapper.sequence_to_one_hot(sequence)
            # one_hot_sequence.shape: (sequence_length, num_nucleobase_letters)

            # flatten sequence matrix to a vector
            flat_one_hot_sequence = torch.flatten(one_hot_sequence)
            # flat_one_hot_sequence.shape: (sequence_length * num_nucleobase_letters,)

            item = flat_one_hot_sequence, coding_value

        elif self.feature_encoding == "label":
            label_encoded_sequence = self.dna_sequence_mapper.sequence_to_label_encoding(
                sequence
            )
            # label_encoded_sequence.shape: (sequence_length,)

            item = label_encoded_sequence, coding_value

        return item


class SuppressSettingWithCopyWarning:
    """
    Suppress SettingWithCopyWarning warning.

    https://stackoverflow.com/a/53954986
    """

    def __init__(self):
        self.original_setting = None

    def __enter__(self):
        self.original_setting = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.original_setting


class AttributeDict(dict):
    """
    Extended dictionary accessible with dot notation.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def generate_dataloaders(configuration):
    """
    Generate training, validation, and test dataloaders from the dataset files.

    Args:
        configuration (AttributeDict): experiment configuration AttributeDict
    Returns:
        tuple containing the training, validation, and test dataloaders
    """
    dataset = DnaSequenceDataset(
        dataset_id=configuration.dataset_id,
        sequence_length=configuration.sequence_length,
        feature_encoding=configuration.feature_encoding,
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
        # pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=configuration.batch_size,
        num_workers=configuration.num_workers,
        # pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configuration.batch_size,
        num_workers=configuration.num_workers,
        # pin_memory=torch.cuda.is_available(),
    )

    return (training_loader, validation_loader, test_loader)


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


def log_pytorch_cuda_info():
    """
    Log PyTorch and CUDA info and device to be used.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.debug(f"{torch.__version__=}")
    logger.debug(f"{DEVICE=}")
    logger.debug(f"{torch.version.cuda=}")
    logger.debug(f"{torch.backends.cudnn.enabled=}")
    logger.debug(f"{torch.cuda.is_available()=}")

    if torch.cuda.is_available():
        logger.debug(f"{torch.cuda.device_count()=}")
        logger.debug(f"{torch.cuda.get_device_properties(DEVICE)}")


if __name__ == "__main__":
    print("this is a module file, import to use")
