import sys
from Bio import SeqIO
import re
import numpy as np


def split_genome(genome="ATCGATATACCA", k=3):
    return re.findall('.'*k, genome)


def genearte_one_genome(genome='ATCGATATACCA', k=3):

    _genome = genome
    _sentence = split_genome(genome=_genome, k=k)

    return _sentence


def fasta2kmers(fasta_file, kmer, out_file):
    '''

    Convert a fasta file into a word/sentence file

    '''

    # traverse the fasta file
    fo = open(out_file + '.sentences', 'w')
    fo2 = open(out_file + '.headers', 'w')

    for record in SeqIO.parse(fasta_file, 'fasta'):
        _genome = str(record.seq).upper()
        sentences = genearte_one_genome(genome=_genome, k=kmer)
        fo.write(" ".join(sentences) + '\n')
        fo2.write(record.description + "\t" + str(len(sentences)) + '\n')
