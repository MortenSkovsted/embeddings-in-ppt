# Readme

This forked version of [Gustav Madslund](https://github.com/gustavmadslund) and [Mikkel Møller Brusen](https://github.com/mikkelbrusen)'s work empliments prediction of multi-labeled datasets of the subcellular localization (subcel). All installation is done as previuse. The original branche take the subcellular localization encoded as an n x 1 array of integers (1-10), where n is the number of proteins. This fork takes its input as an n x 10 array of intergers (0,1) where 0/1 indicate the presents/absences of a perticular location. E.g. Old format: 2 -> new format: [0,1,0,0,0,0,0,0,0,0].
Additionally error function was changed to accommodate this new option for multi-labelling, and the output metrics was changed to output F1 and Exact match as these are more meaningful metrics when multi-labels are possible.


secpred part of the code have been left untoched and untested.
forked by [MortenSkovsted](https://github.com/MortenSkovsted)

This is the code repository that accompany the master thesis by [Gustav Madslund](https://github.com/gustavmadslund) and [Mikkel Møller Brusen](https://github.com/mikkelbrusen).

The goal of the project was to evaluate pre-trained amino acid embeddings in protein prediction tasks



## Software Requirements

The software is coded in Python 3.6 using the Pytorch 1.1 version. 
To run the software smoothly, it is recommended to use those versions.

The code was made to be run on a CUDA GPU but can run on CPU too, although this will take forever... 
In order to run configurations that utilize the bi-direction pre-trained embeddings, at least 17GB RAM.

## Setup & Data

In order to run all configurations the following datasets are needed:

+ **Deeploc** which is the deeploc dataset encoded as profiles
+ **Deeploc_raw** which is the deeploc dataset without encoding (raw sequences)
+ **SecPred** which is the filtered CullPDB dataset encoded as profiles. Files with _no_x have X replaced by A.   
+ **SecPred_raw** which is the CB513 dataset without encoding (raw sequences). X has been replaced by A.

all of which can be [downloaded here](https://drive.google.com/drive/folders/1-qPOetLSYrrlFvcjmt2lSAwKoR-_AXFm?usp=sharing)
See comment in the top!!!

The datasets should then be positioned in the [`data/` directory](data/) similarly to the already included Deeploc_raw dataset.

## Training models

Model architecture and other settings are controlled by config files in the [`configs/{task}` directory](configs/). Each config is task specific, such that subcellular localization configurations can be found in [`configs/subcel` directory](configs/subcel/) and secondary structure prediction configurations can be found in [`configs/secpred` directory](configs/secpred/).

To start training a model, we need to first give the task as argument when running `main.py` e.g. `subcel` and then choose a configuration with `--config`. For example, if we want to train with the configuration `configs/subcel/deeploc_raw`, we should use the following command:

    python3 main.py subcel --config deeploc_raw

All configurations are created such that no hyperparameters needs to be specified, although they are possible if you want to do experiments with a specific configuration. For a list of all avaiable arguments the following commands are usefull:

    python3 main.py --help
    python3 main.py subcel --help
    python3 main.py secpred --help

The best models based on validation performance will be saved under `save/{task}/{config_name}/` where task can be subcel or secpred and config_name is the configuration that is training.
