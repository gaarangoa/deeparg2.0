import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from time import time
from dataset import obtain_labels, obtain_dataset_wordvectors, obtain_dataset_alignments
from model import DeepARG
import numpy as np
from sklearn.model_selection import train_test_split
import json


# tensorboard
tensorboard = TensorBoard(log_dir="../logs/{}".format(time()))

# load dataset
classes, groups, index, group_labels, classes_labels = obtain_labels(
    labels_file='../test/dataset.no_centroids.ss.headers'
)

dataset_wordvectors = obtain_dataset_wordvectors(
    dataset_file='../test/dataset.no_centroids.ss.sv',
    labels_file='../test/dataset.no_centroids.ss.headers'
)

reverse_classes_dict = {int(classes[i]): i for i in classes}
reverse_groups_dict = {int(groups[i]): i for i in groups}
dataset_index = np.array([ix for ix, i in enumerate(dataset_wordvectors)])

train_dataset_wordvectors, test_dataset_wordvectors, target_train, target_val = train_test_split(
    dataset_wordvectors,
    dataset_index,
    test_size=0.2
)

train_labels_class = classes_labels[target_train]
train_labels_group = group_labels[target_train]
test_labels_class = classes_labels[target_val]
test_labels_group = group_labels[target_val]

deeparg = DeepARG(
    input_dataset_wordvectors_size=dataset_wordvectors.shape[1],
    classes_labels=classes_labels,
    group_labels=group_labels,
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
    },
    {
        'arg_class_output': train_labels_class,
        'arg_group_output': train_labels_group
    },
    epochs=50,
    batch_size=32,
    validation_split=0.2,
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
    open('../model/deearg2.parameters.json', 'w')
)

model.save('../model/deeparg2.h5')
