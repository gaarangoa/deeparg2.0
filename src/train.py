import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from time import time
from dataset import obtain_labels, obtain_dataset
from model import DeepARG


# load dataset
classes, groups, index, group_labels, classes_labels = obtain_labels(
    labels_file='../database/dataset.ss.headers'
)

dataset = obtain_dataset(
    dataset_file='../database/dataset.sv.txt',
    index=index
)

train_dataset = dataset[:100]
train_labels_class = classes_labels[:100]
train_labels_group = group_labels[:100]

deeparg = DeepARG(
    dataset=dataset,
    classes_labels=classes_labels,
    group_labels=group_labels,
    classes=classes,
    groups=groups
)

model = deeparg.model()

model.compile(
    optimizer='adam',
    loss={
        'arg_group_output': 'categorical_crossentropy',
        'arg_class_output': 'categorical_crossentropy'
    },
    loss_weights={
        'arg_group_output': 1.0,
        'arg_class_output': 1.0
    }
)

# Add tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# And trained it via:
model.fit(
    {
        'wordvectors_input': train_dataset,
        # 'aux_input': additional_data
    },
    {
        'arg_class_output': train_labels_class,
        'arg_group_output': train_labels_group
    },
    epochs=50,
    batch_size=32,
    callbacks=[tensorboard]
)

# /Library/Frameworks/Python.framework/Versions/3.6/bin/tensorboard --logdir=src/logs/
