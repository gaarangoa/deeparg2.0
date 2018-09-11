from tensorflow import keras
from DeepNovelARG.dataset import obtain_labels, obtain_dataset_wordvectors, obtain_dataset_alignments
import json
import click
from GeneTools.fasta2kmers import fasta2kmers
import os
from tqdm import tqdm
import logging


@click.command()
@click.option('--inputfile', required=True, help='input fasta file')
@click.option('--modeldir', default='', required=True, help='directory where the model was downloaded')
@click.option('--outdir', default='', required=True, help='output directory where to store the results')
@click.option('--minp', default=0.1, help='minimum probability')
@click.option('--kmer', default=11, help='kmer length (default: 11)')
def predict(inputfile, modeldir, outdir, kmer, minp):
    """

    Input a fasta file and predict the ARG-like sequences.

    """

    logging.basicConfig(
        filename=outdir + '/log',
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s - %(message)s"
    )

    log = logging.getLogger()

    log.info(" Convert input file into a kmer's sentence file: Initiated")
    # Convert fasta file to kmer's sentence file
    # produces: input.kmers.tsv.sentences
    #           input.kmers.tsv.headers

    fasta2kmers(inputfile, kmer, outdir + '/input.kmers.tsv')
    log.info(" Convert input file into a kmer's sentence file: Finished")

    # Get sentence vectors using fasttext
    # produces: input.kmers.tsv.sentences.wv

    log.info('Get sentence vectors using fasttext: Initiated')
    os.system(
        'fasttext print-sentence-vectors ' +
        modeldir + '/model.bin < ' +
        outdir + '/input.kmers.tsv.sentences > ' +
        outdir + '/input.kmers.tsv.sentences.wv '
    )
    log.info('Get sentence vectors using fasttext: Finished')

    # load dataset
    log.info('Loading dataset for classification')
    dataset_wordvectors = obtain_dataset_wordvectors(
        dataset_file=outdir+'/input.kmers.tsv.sentences.wv',
        labels_file=outdir+'/input.kmers.tsv.headers',
    )

    # load deep learning model
    log.info('Loading Deep Neural Network model')
    model = keras.models.load_model(modeldir+'/deeparg2.h5')
    ynew = model.predict(
        {
            'wordvectors_input': dataset_wordvectors,
        },
    )

    log.info("Loading Neural Network metadata")
    # load metadata from the trained model
    metadata = json.load(open(modeldir+'/deearg2.parameters.json'))

    # load file that contains the order in which the sequences are processed
    log.info(
        "Loading file *.headers that contains the order in which each entry apears")
    file_order = [
        i.strip().split("\t")[0] for i in open(outdir+'/input.kmers.tsv.headers')
    ]

    # write output files
    log.info("Write results for classes annotation")
    fo = open(outdir + '/predicted.classes.txt', 'w')
    fo.write("#Query\tProbability\tPrediction\n")
    for _ix in tqdm(range(len(ynew[0]))):
        y_pred = ynew[0][_ix]
        query = file_order[_ix]
        for ix, i in enumerate(y_pred):
            if round(i, 2) <= minp:
                continue
            fo.write("\t".join([
                query,
                str(round(i, 2)),
                metadata['reverse_classes_dict'][str(ix)]
            ]) + "\n")

    log.info("Write results for groups/genes annotation")
    fo = open(outdir + '/predicted.groups.txt', 'w')
    fo.write("#Query\tProbability\tPrediction\n")
    for _ix in tqdm(range(len(ynew[1]))):
        y_pred = ynew[1][_ix]
        query = file_order[_ix]
        for ix, i in enumerate(y_pred):
            if round(i, 2) <= minp:
                continue
            fo.write("\t".join([
                query,
                str(round(i, 2)),
                metadata['reverse_groups_dict'][str(ix)]
            ])+"\n")