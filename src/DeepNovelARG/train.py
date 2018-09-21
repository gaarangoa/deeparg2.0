import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from time import time
from DeepNovelARG.dataset import obtain_labels, obtain_dataset_wordvectors, obtain_dataset_alignments, obtain_test_labels
from DeepNovelARG.model import DeepARG
import numpy as np
from sklearn.model_selection import train_test_split
import json
import click
import tensorflow.contrib.eager as tfe
import logging
import sys
import os


@click.command()
@click.option('--inputdir', required=True, help='input directory with the output from fasta2vec')
@click.option('--outdir', default='', required=True, help='output directory where to store the model')
@click.option('--epoch', default=10, required=False, help='number of epochs to run the model [default 10]')
@click.option('--batch', default=32, required=False, help='batch size for using during training [default 32]')
@click.option('--maxlen-conv', default=1500, required=False, help='max sequence length to consider for convolutional network [default 1500]')
@click.option('--prefix', default="", required=False, help='prefix used during training for tensorboard. Useful for keeping track of different sessions')
def train(inputdir, outdir, epoch, batch, maxlen_conv, prefix):
    '''
        Train a the deepARG+ architecture (convolutional network + word vectors deep network) for the prediciton
        of categories (antibiotics) and groups (gene names).

        Although, the topology was tested for ARGs, it can be used for any dataset. For details,
        please take a look at: git

    '''

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    log_file = logging.FileHandler(filename=outdir + '/train.log',)
    log_stdout = logging.StreamHandler(sys.stdout)
    handlers = [log_file, log_stdout]

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s - %(message)s",
        handlers=handlers
    )

    log = logging.getLogger()

    # what device is using?
    if tfe.num_gpus() <= 0:
        log.info("Using CPU: training may take a while!")
    else:
        log.info("Using GPU: training may take a while")

    # tensorboard
    name = f'deepARG+_{prefix}_{time()}'
    log.info("starting TensorBoard")
    tensorboard = TensorBoard(log_dir=outdir+f'/logs/{name}')

    # Model Checkpoint
    ckpt_file = 'model.{epoch:03d}.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(ckpt_file, verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)


    # load training dataset wordvectors
    log.info('loading training labels')
    classes, groups, index, train_group_labels, train_class_labels = obtain_labels(
        labels_file=inputdir+'/train.input.kmers.tsv.headers'
    )

    log.info("adding testing labels")
    test_group_labels, test_class_labels = obtain_test_labels(
        classes=classes,
        groups=groups,
        labels_file=inputdir+'/test.input.kmers.tsv.headers'
    )

    log.info("Loading training dataset: wordvectors and numerical signals")
    train_dataset_wordvectors, train_dataset_numerical = obtain_dataset_wordvectors(
        dataset_file=inputdir+'/train.input.kmers.tsv.sentences.wv',
        labels_file=inputdir + '/train.input.kmers.tsv.headers',
        maxlen=maxlen_conv
    )

    reverse_classes_dict = {int(classes[i]): i for i in classes}
    reverse_groups_dict = {int(groups[i]): i for i in groups}

    log.info("Loading testing dataset: ")
    test_dataset_wordvectors, test_dataset_numerical = obtain_dataset_wordvectors(
        dataset_file=inputdir+'/test.input.kmers.tsv.sentences.wv',
        labels_file=inputdir + '/test.input.kmers.tsv.headers',
        maxlen=maxlen_conv
    )

    log.info('loading deep learning model')
    deeparg = DeepARG(
        input_dataset_wordvectors_size=train_dataset_wordvectors.shape[1],
        input_convolutional_dataset_size=maxlen_conv,
        num_classes=len(classes),
        num_groups=len(groups)
    )

    model = deeparg.model()

    log.info('compiling deep learning model deepARG+')

    model.compile(
        optimizer='adam',
        loss={
            'arg_group_output': 'binary_crossentropy',
            'arg_class_output': 'binary_crossentropy'
        },
        loss_weights={
            'arg_group_output': 1.0,
            'arg_class_output': 1.0
        },
        metrics=['accuracy']
    )

    log.info("Training deepARG+")
    # And trained it via:
    model.fit(
        {
            'wordvectors_input': train_dataset_wordvectors,
            'convolutional_input': train_dataset_numerical
        },
        {
            'arg_class_output': train_class_labels,
            'arg_group_output': train_group_labels
        },
        epochs=epoch,
        batch_size=batch,
        validation_data=(
            {
                'wordvectors_input': test_dataset_wordvectors,
                'convolutional_input': test_dataset_numerical
            },
            {
                'arg_class_output': test_class_labels,
                'arg_group_output': test_group_labels
            }
        ),
        callbacks=[tensorboard, checkpoint],
        shuffle=True
    )

    log.info("Storing deepARG+ metadata")
    json.dump(
        {
            'optimizer': 'adam',
            'epochs': epoch,
            'classes_dict': classes,
            'groups_dict': groups,
            'reverse_classes_dict': reverse_classes_dict,
            'reverse_groups_dict': reverse_groups_dict
        },
        open(outdir+'/deeparg2.parameters.json', 'w')
    )

    # log.info("Storing trained model after "+str(epoch)+" epochs.")
    # model.save(outdir+'/deeparg2.h5')
