from tensorflow import keras
from DeepNovelARG.dataset import obtain_labels, obtain_dataset_wordvectors, obtain_dataset_alignments
import json
import click
from DeepNovelARG.fasta2kmers import fasta2kmers
import os
from tqdm import tqdm
import logging
import sys


@click.command()
@click.option('--inputfile', required=True, help='input fasta file')
@click.option('--wordvec-model', default='', required=True, help='wordvector model [model.bin]')
@click.option('--outdir', default='', required=True, help='output directory where to store the results')
@click.option('--kmer', default=11, help='kmer length [default: 11]')
@click.option('--prefix', default='', help='prefix to add to output files [train, test]')
def fasta2vec(inputfile, wordvec_model, outdir, kmer, prefix):
    """

    Input a fasta file and builds the wordvector files

    """

    log_file = logging.FileHandler(
        filename=outdir + '/' + prefix + '.fasta2vec.log'
    )
    log_stdout = logging.StreamHandler(sys.stdout)
    handlers = [log_file, log_stdout]

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s - %(message)s",
        handlers=handlers
    )

    log = logging.getLogger()

    log.info("Convert input file into a kmer's sentence file: Initiated")
    # Convert fasta file to kmer's sentence file
    # produces: input.kmers.tsv.sentences
    #           input.kmers.tsv.headers

    fasta2kmers(inputfile, kmer, outdir + '/' + prefix + '.input.kmers.tsv')
    log.info("Convert input file into a kmer's sentence file: Finished")

    # Get sentence vectors using fasttext
    # produces: input.kmers.tsv.sentences.wv

    log.info('Get sentence vectors using fasttext: Initiated')
    os.system(
        'fasttext print-sentence-vectors ' +
        wordvec_model + ' < ' +
        outdir + '/' + prefix + '.input.kmers.tsv.sentences > ' +
        outdir + '/' + prefix + '.input.kmers.tsv.sentences.wv '
    )
    log.info('Get sentence vectors using fasttext: Finished')

    # TODO: postprocess sequences to get only vectors in HDF5 files
