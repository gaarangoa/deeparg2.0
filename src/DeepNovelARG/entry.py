import click
from DeepNovelARG.predict import predict
from DeepNovelARG.fasta2vec import fasta2vec
from DeepNovelARG.train import train


@click.group()
def cli():
    '''
        novelDeepARG:  A deep learning approach for the identification of novel ARGs.
        Author(s):   Gustavo Arango (gustavo1@vt.edu)
    '''
    pass


cli.add_command(predict)
cli.add_command(fasta2vec)
cli.add_command(train)
