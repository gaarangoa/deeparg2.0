# DNN-ARG plus v2.0
This is an experimental version of deepARG, please see https://github.com/gaarangoa/deeparg 
## Train

DeepARG+ has been released in a docker image to avoid library issues and conflict with newer versions of the libraries.


    docker run --runtime=nvidia -it -v $PWD:/data/  --rm gaarangoa/deepargplus:latest deepARG+ train \
        --inputdir /data/ \
        --outdir /data/ \
        --prefix DL \
        --epoch 10 \
        --batch 32


    deepARG+ train         \
        --inputdir ./         \
        --outdir ./         \
        --prefix DL         \
        --epoch 2    \
        --batch 32

## Predict

    docker run --runtime=nvidia -it -v $PWD:/data/  --rm gaarangoa/deepargplus:latest deepARG+  predict \
        --inputfile /data/tests/b.fasta \
        --wordvec-model /data/wvecmodel/model.bin \
        --deeparg-model /data/DL.001.hdf5 \
        --deeparg-parameters /data/DL.parameters.json \
        --outdir /data/tests/ \
        --prefix bla


        deepARG+ predict --inputfile b.fa --wordvec-model ../wvecmodel/model.bin --deeparg-model ../DL.001.hdf5 --deeparg-parameters ../DL.parameters.json --outdir ./ --prefix bla

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
