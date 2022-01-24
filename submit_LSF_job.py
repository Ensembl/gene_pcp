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
Submit an LSF job to train or test a neural network protein coding potential classifier.
"""


# standard library imports
import argparse
import datetime as dt
import pathlib
import subprocess
import sys

# third party imports
import yaml

# project imports
from protein_coding_potential_mlp import ProteinCodingClassifier
from utils import AttributeDict


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--configuration",
        help="path to the experiment configuration YAML file",
    )
    argument_parser.add_argument(
        "--mem_limit",
        default=16384,
        type=int,
        help="memory limit for all the processes that belong to the job",
    )
    argument_parser.add_argument(
        "--gpu",
        action="store_true",
        help="submit training job to the gpu queue",
    )
    argument_parser.add_argument(
        "--checkpoint",
        help="path to the saved experiment checkpoint",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")

    args = argument_parser.parse_args()

    # submit new classifier training
    if args.configuration:
        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)
        configuration = AttributeDict(configuration)

        configuration.datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        dataset_id = configuration.dataset_id

        job_name = f"{configuration.experiment_prefix}_{configuration.dataset_id}_{configuration.datetime}"
        root_directory = configuration.save_directory

        pipeline_command_elements = [
            "python protein_coding_potential_mlp.py",
            f"--datetime {configuration.datetime}",
            f"--configuration {args.configuration}",
            "--train",
            "--test",
        ]

    # test a trained classifier
    elif args.test and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        network = ProteinCodingClassifier.load_from_checkpoint(checkpoint_path)
        configuration = network.hparams

        dataset_id = configuration.dataset_id

        job_name = checkpoint_path.stem
        root_directory = checkpoint_path.parent

        pipeline_command_elements = [
            "python protein_coding_potential_mlp.py",
            f"--checkpoint {args.checkpoint}",
            "--test",
        ]

    # no task specified
    else:
        print(__doc__)
        argument_parser.print_help()
        sys.exit()

    pipeline_command = " ".join(pipeline_command_elements)

    # specify lower mem_limit for dev datasets jobs
    dataset_ids_mem_limit = {"1pct": 1024, "5pct": 2048, "20pct": 8192}

    mem_limit = dataset_ids_mem_limit.get(dataset_id, args.mem_limit)

    # common job arguments
    bsub_command_elements = [
        "bsub",
        f"-M {mem_limit}",
        f"-o {root_directory}/{job_name}/stdout.log",
        f"-e {root_directory}/{job_name}/stderr.log",
    ]

    if args.gpu:
        num_gpus = 1
        gpu_memory = 16384  # 16 GiBs
        # gpu_memory = 32510  # ~32 GiBs, total Tesla V100 memory

        bsub_command_elements.extend(
            [
                "-q gpu",
                f'-gpu "num={num_gpus}:gmem={gpu_memory}:j_exclusive=yes"',
                f"-M {mem_limit}",
                f'-R"select[mem>{mem_limit}] rusage[mem={mem_limit}] span[hosts=1]"',
            ]
        )
    else:
        bsub_command_elements.extend(
            [
                "-q production",
                f'-R"select[mem>{mem_limit}] rusage[mem={mem_limit}]"',
            ]
        )

    bsub_command_elements.append(pipeline_command)

    bsub_command = " ".join(bsub_command_elements)
    print(f"running command:\n{bsub_command}")

    subprocess.run(bsub_command, shell=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")