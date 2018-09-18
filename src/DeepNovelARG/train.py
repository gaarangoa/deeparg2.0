import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from time import time
from DeepNovelARG.dataset import obtain_labels, obtain_dataset_wordvectors, obtain_dataset_alignments
from DeepNovelARG.model import DeepARG
import numpy as np
from sklearn.model_selection import train_test_split
import json
import click
import tensorflow.contrib.eager as tfe


@click.command()
@click.option('--inputdir', required=True, help='input directory with the output from fasta2vec')
@click.option('--outdir', default='', required=True, help='output directory where to store the model')
@click.option('--epoch', default=10, required=False, help='number of epochs to run the model [default 10]')
@click.option('--ptrain', default=0.3, required=False, help='fraction of the dataset used for training [default 0.3]')
@click.option('--batch', default=32, required=False, help='batch size for using during training [default 32]')
def train(inputdir, outdir, epoch, ptrain, batch):

    # what device is using?
    if tfe.num_gpus() <= 0:
        print("Using CPU")
    else:
        print("Using GPU")

    # tensorboard
    tensorboard = TensorBoard(log_dir=outdir+"/logs/{}".format(time()))

    # load training dataset wordvectors
    classes, groups, index, train_group_labels, train_class_labels = obtain_labels(
        labels_file=inputdir+'/input.kmers.tsv.headers'
    )

    train_dataset_wordvectors, train_dataset_numerical = obtain_dataset_wordvectors(
        dataset_file=inputdir+'/input.kmers.tsv.sentences.wv',
        labels_file=inputdir+'/input.kmers.tsv.headers'
    )

    reverse_classes_dict = {int(classes[i]): i for i in classes}
    reverse_groups_dict = {int(groups[i]): i for i in groups}
    dataset_index = np.array(
        [ix for ix, i in enumerate(train_dataset_wordvectors)]
    )

    deeparg = DeepARG(
        input_dataset_wordvectors_size=train_dataset_wordvectors.shape[1],
        input_convolutional_dataset_size=1500,
        classes_labels=train_class_labels,
        group_labels=train_group_labels,
        classes=classes,
        groups=groups
    )

    model = deeparg.model()

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
        validation_split=ptrain,
        callbacks=[tensorboard],
        shuffle=True
    )

    json.dump(
        {
            'classes_dict': classes,
            'groups_dict': groups,
            'reverse_classes_dict': reverse_classes_dict,
            'reverse_groups_dict': reverse_groups_dict
        },
        open(outdir+'/deeparg2.parameters.json', 'w')
    )

    model.save(outdir+'/deeparg2.h5')
