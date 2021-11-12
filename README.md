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

Specify parameters and hyperparameters for your experiment by editing or copying the `experiment.yaml` configuration file.

### train and test a classifier

run training on a compute node
```
python protein_coding_potential.py --train --test -ex <experiment settings YAML file path>

# e.g.
python protein_coding_potential.py --train --test -ex experiment.yaml
```


## License

[Apache License 2.0](LICENSE)
