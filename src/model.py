import tensorflow as tf
from tensorflow import keras


class DeepARG():
    def __init__(self, dataset=[], classes_labels=[], group_labels=[], classes=[], groups=[]):
        # setup parameters for model
        self.embedding_size = dataset.shape[1]
        self.total_arg_classes = len(classes)
        self.total_arg_groups = len(groups)

    def model(self):
        # Input layer
        wordvectors_input = keras.Input(
            shape=(self.embedding_size,),
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
            self.total_arg_groups,
            activation="sigmoid",
            name="arg_group_output"
        )(wordvectors_nn_3)

        # arg classes (antibiotics)
        arg_class_output = keras.layers.Dense(
            self.total_arg_classes,
            activation="sigmoid",
            name="arg_class_output"
        )(wordvectors_nn_3)

        # Topology of the model
        _model = keras.models.Model(
            inputs=[wordvectors_input],
            outputs=[
                arg_class_output,
                arg_groups_output
            ]
        )

        return _model
