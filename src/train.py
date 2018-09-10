import tensorflow as tf
from tensorflow import keras

from dataset import obtain_labels, obtain_dataset

classes, groups, index, group_labels, classes_labels = obtain_labels(
    labels_file='../database/dataset.ss.headers'
)

dataset = obtain_dataset(
    dataset_file='../database/dataset.sv.txt',
    index=index
)

embedding_size = 512
total_arg_classes = len(classes)
total_arg_groups = len(groups)

# Input layer
wordvectors_input = keras.Input(
    shape=(embedding_size,),
    name="wordvectors_input"
)

# Hiden Layer
wordvectors_nn_1 = keras.layers.Dense(
    800,
    activation='relu'
)(wordvectors_input)

wordvectors_nn_2 = keras.layers.Dense(
    600,
    activation='relu'
)(wordvectors_nn_1)

wordvectors_nn_3 = keras.layers.Dense(
    400,
    activation='relu'
)(wordvectors_nn_2)

# Output layers
# arg groups (names)
arg_groups_output = keras.layers.Dense(
    total_arg_groups,
    activation="sigmoid",
    name="arg_group_output"
)(wordvectors_nn_2)

# arg classes (antibiotics)
arg_class_output = keras.layers.Dense(
    total_arg_classes,
    activation="sigmoid",
    name="arg_class_output"
)(wordvectors_nn_2)

# Topology of the model
model = keras.models.Model(
    inputs=[wordvectors_input],
    outputs=[
        arg_class_output,
        arg_groups_output
    ]
)

# model.compile(optimizer='adam', loss='categorical_crossentropy')

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

# And trained it via:
model.fit(
    {
        'wordvectors_input': headline_data,
        # 'aux_input': additional_data
    },
    {
        'arg_class_output': labels,
        'arg_group_output': labels
    },
    epochs=50,
    batch_size=32
)
