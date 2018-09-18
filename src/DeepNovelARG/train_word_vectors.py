import re
from Bio import SeqIO
import numpy as np
import os
import h5py

from tqdm import tqdm
import logging

import click


def split_genome(genome="ATCGATATACCA", k=3):
    return re.findall('.'*k, genome)


def genearte_genomes(genome='ATCGATATACCA', k=3, words=50):
    sentences = []
    for index in range(0, k):
        _genome = genome[index:]
        _sentence = split_genome(genome=_genome, k=k)
        _fraction = int(len(genome) / k) - len(_sentence)

        if _fraction > 0:
            _sentence.append('')

        sentences.append(np.array(_sentence, dtype="U"))

    return np.array(sentences)


def genome_to_doc(input_file="", kmer=16, label="", f5=""):
    ''' This function transforms a sequence genome to a document of kmers '''

    records = []
    for record in SeqIO.parse(input_file, 'fasta'):
        _genome = str(record.seq).upper()
        _kmer_count = int(len(_genome) / kmer)
        records.append({
            'sentences': genearte_genomes(genome=_genome, k=kmer),
            'id': record.id,
            '_kmer_count': _kmer_count,
            'label': label
        })

    return records


@click.command()
@click.option('--inputfile', required=True, help='input fasta file')
@click.option('--modeldir', default='', required=True, help='directory where the word2vec model was downloaded')
@click.option('--outdir', default='', required=True, help='output directory where to store the results')
@click.option('--kmer', default=11, help='kmer length [default: 11]')
def train_word_vectors(inputfile, modeldir, outdir, kmer):
    '''
    train word vectors using fasttext.

    the input fasta file --inputfile is splitted into consecutive kmers
    of lenth --kmer. Thus, there are a total of --kmer versions of
    the same sequence.

    Once all "sentence" are generated, they are trained using the skipgram
    model from fasttext to generate a word vectors representation.

    '''
    pass
