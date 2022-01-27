# Protein Coding Potential

Gene sequence protein coding potential classification with a Machine Learning approach.

More information and background for the project:
https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/Protein+Coding+Potential


## project setup

### dev environment setup

set up a Python virtual environment for the project and install its dependencies
```
pyenv install 3.9.7

pyenv virtualenv 3.9.7 protein_coding_potential

poetry install
```

### dataset

generate full and dev dataset pickled dataframes
```
python dataset_generation.py --generate_datasets --coding_transcripts_path <coding transcripts FASTA path> --non_coding_transcripts_path <non-coding transcripts FASTA path>
```

### experiment setup

Specify parameters and hyperparameters for your experiment by editing or copying the `configuration.yaml` configuration file.

### train a classifier

run training directly
```
python protein_coding_potential.py --train --test --configuration <experiment configuration YAML file path>

# e.g.
python protein_coding_potential.py --train --test --configuration configuration.yaml
```

submit a training job on LSF
```
python submit_LSF_job.py --pipeline <pipeline script path> --configuration <experiment configuration file path>

# e.g.
python submit_LSF_job.py --pipeline protein_coding_potential.py --configuration configuration.yaml

python submit_LSF_job.py --pipeline protein_coding_potential_mlp.py --configuration configuration_mlp.yaml
```

### test a classifier

load a checkpoint and run testing directly
```
python protein_coding_potential.py --test --checkpoint <checkpoint path>
```

submit a testing job on LSF
```
python submit_LSF_job.py --pipeline <pipeline script path> --checkpoint <checkpoint path>
```

## License

[Apache License 2.0](LICENSE)
