# DeepARG v2.0
This repository contains the update of deepARG (deep learning based approach for antibiotic resistance gene annotation)

## Requierements

* Python3.6

### Install fasttext

    wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
    unzip v0.1.0.zip
    cd fastText-0.1.0
    make

### Download model to local machine

    wget https://bench.cs.vt.edu/ftp/data/gustavo1/novelDeepARG/model.gz
    gunzip model.gz


## Install

    git clone https://github.com/gaarangoa/deeparg2.0.git
    cd deeparg2.0
    pip3 install . --upgrade --user

## Training

You need to make sure that your fasta file header follows this schema:

    >gene_id|arg_category|arg_name|arg_group

arg_name refers to the name of the arg e.g., OXA-1
arg_group refers to the grouping of very similar args, for instance OXA


* First convert the fasta file to a word vector representation

        deepARG+ fasta2vec --help

Check the log file to make sure the script ran without problems.

* Second, run the training

        deepARG+ train --help

Thus, the model file will be generated


## Usage
