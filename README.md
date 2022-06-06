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
python dataset_generation.py --generate_datasets --coding_transcripts <coding transcripts FASTA path> --non_coding_transcripts <non-coding transcripts FASTA path>
```

### experiment setup

Specify parameters and hyperparameters for your experiment by editing or copying one of the configuration YAML files.

### train a classifier

run training directly
```
python <pipeline script> --train --configuration <experiment configuration file>

# e.g.
python transformer_pipeline.py --train --configuration transformer_configuration.yaml
```

submit a training job on LSF
```
python submit_LSF_job.py --pipeline <pipeline script> --configuration <experiment configuration file>

# e.g.
python submit_LSF_job.py --pipeline transformer_pipeline.py --configuration transformer_configuration.yaml
```

### test a classifier

load a checkpoint and run testing directly
```
python <pipeline script> --test --checkpoint <checkpoint path>
```

submit a testing job on LSF
```
python submit_LSF_job.py --pipeline <pipeline script> --checkpoint <checkpoint path>
```


## License

[Apache License 2.0](LICENSE)
