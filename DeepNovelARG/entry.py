import click
from DeepNovelARG.predict import predict


@click.group()
def cli():
    '''
        Deep Novel ARG:  A deep learning approach for the identification of novel ARGs.

        Author(s):   Gustavo Arango (gustavo1@vt.edu)

        Usage:       dnovelarg --help
    '''
    pass


cli.add_command(predict)
