# Multi-clue reconstruction of sharing chains for social media images [[arXiv]](http://arxiv.org/abs/2108.02515)

## Sebastiano Verde, Cecilia Pasquini, Federica Lago, Alessandro Goller, Francesco GB De Natale, Alessandro Piva, and Giulia Boato 

## Abstract

The amount of multimedia content shared everyday, combined with the level of realism reached by recent fake-generating technologies, threatens to impair the trustworthiness of online information sources. The process of uploading and sharing data tends to hinder standard media forensic analyses, since multiple re-sharing steps progressively hide the traces of past manipulations. At the same time though, new traces are introduced by the platforms themselves, enabling the reconstruction of the sharing history of digital objects, with possible applications in information flow monitoring and source identification. In this work, we propose a supervised framework for the reconstruction of image sharing chains on social media platforms. The system is structured as a cascade of backtracking blocks, each of them tracing back one step of the sharing chain at a time. Blocks are designed as ensembles of classifiers trained to analyse the input image independently from one another by leveraging different feature representations that describe both content and container of the media object. Individual decisions are then properly combined by a late fusion strategy. Results highlight the advantages of employing multiple clues, which allow accurately tracing back up to three steps along the sharing chain.

## Prerequisites

You need python >=3.6, pip is advised for packet manager.
Matlab version R2018b, or compatible.

To install matlab engine for python, after the activation of the virtual environment, do the following (`matlabroot` on Linux is typically `usr/local/MATLAB/Rxxxxx/`)

Please also install ``libimage-exiftool-perl`` using the system packet manager

```
cd matlabroot/extern/engines/python
<path-to-venv>/bin/python setup.py install
```
WARNING: the second step might require root

## Structure


```
project
├───dat
├───dataset
│   ├───dataset_name1
│   │   ├───original
│   │   ├───SN_1
│   │   ├───SN_2
│   │   ├───...
│   │   └───SN_N
│   └───...
├───features
└───src
    └───support
        ├───jpegtbx_1.4
        └───ZIG_ZIG_SCAN

```

`dat` folder is meant to contain the train/test/validation splits (see *Generate splits* for more details)

Inside the `dataset` folder you can download the selected dataset(s) if you don't already have the extracted features. For instance you can download:
* R-SMUD http://loki.disi.unitn.it/~rvsmud/ (_controlled_ from now on)
* V-SMUD http://loki.disi.unitn.it/~rvsmud/ (_uncontrolled_ from now on)
* etc.

`features` contains HDF5 files containing train, test a (optionally) validation feature vectors and labels.
It can also contain DCT, META and HEADER feature files.

## Getting Started

You can find extracted features and labels [here](https://drive.google.com/file/d/1t5gjDJdeFZeYvxR97NIKzWjGK3JNjy-G/view?usp=sharing).

Otherwise you can genereate the features and labels with the following steps.

### Generate splits

To populate `dat` use

```
python generate-dat.py -db <name_of_db> -db_config <number_of_config>

```

### Generate META, DCT and/or HEADER features from dat file


To populate `folder`, which contais metadata and DCT features extracted with MATLAB use  `feature_extraction.m`


MATLAB scripts will allow you to specify the selected parameters through the user input, namely
* the dataset: valid options for the `dataset` field are {controlled, uncontrolled, iplab, iplab_irene, isima, public, ucid}, but custom dataset can be added
* the configuration: can be equal to {1,2,3} and correspond to the testing configuration as specified in the paper (i.e., the number of compressions)
* the features to extract: metadata features from JPEG or DCT histograms or HEADER features or all of them.


### Generate h5py

To generate the h5py containing labels and features for a datast use:

```
python feature_extraction.py -db <name_of_db> -db_config <number_of_config>

```

## Using the cascade code

To train and test the classifiers and get the results in the `results` folder for the cascade approach run the following command. Additional parameters can be set see `--help` for details.

```
python bks-cascade.py
```
for the informed version, which stops as soon as it reaches TW in the chain reconstruction process use

```
python bks-informed-cascade.py
```

## Acknoledgment

This code is partially based on Quoc-Tin Phan's work [1].

We thank Prof. Fabio Roli (University of Cagliari, Italy) for the valuable insights on classifier fusion and the BKS method. We also thank Chiara Albisani (University of Florence, Italy) for contributing to the parsing of header data. 

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00112090136 and by the PREMIER project, funded by the Italian Ministry of Education, University, and Research (MIUR).


## Citation

If you use this code in your work, please cite our paper:


```
@misc{X,
      title={Multi-clue reconstruction of sharing chains for social media images}, 
      author={Sebastiano Verde and Federica Lago and Cecilia Pasquini and Giulia Boato and Francesco GB De Natale and Alessandro Piva},
      year={2021},
}
```

## Bibliography

[1] Phan, Quoc-Tin, et al. *"Tracking multiple image sharing on social networks." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2019.