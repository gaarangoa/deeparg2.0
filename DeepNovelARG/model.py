import tensorflow as tf
from tensorflow import keras


class DeepARG():
    def __init__(self, input_dataset_wordvectors_size=0, classes_labels=[], group_labels=[], classes=[], groups=[]):
        # setup parameters for model
        self.input_dataset_wordvectors_size = input_dataset_wordvectors_size
        self.total_arg_classes = len(classes)
        self.total_arg_groups = len(groups)

    def model(self):
        # Input layer
        wordvectors_input = keras.Input(
            shape=(self.input_dataset_wordvectors_size,),
            name="wordvectors_input"
        )

        # Hiden Layer
        wordvectors_nn_1 = keras.layers.Dense(
            800,
            activation='relu'
        )(wordvectors_input)

        wordvectors_dropout_1 = keras.layers.Dropout(0.2)(wordvectors_nn_1)

        wordvectors_nn_2 = keras.layers.Dense(
            600,
            activation='relu'
        )(wordvectors_dropout_1)

        wordvectors_dropout_2 = keras.layers.Dropout(0.2)(wordvectors_nn_2)

        wordvectors_nn_3 = keras.layers.Dense(
            400,
            activation='relu'
        )(wordvectors_dropout_2)

        wordvectors_dropout_3 = keras.layers.Dropout(0.2)(wordvectors_nn_3)

        wordvectors_nn_4 = keras.layers.Dense(
            200,
            activation='relu'
        )(wordvectors_dropout_3)

        # Output layers
        # arg groups (names)
        arg_groups_output = keras.layers.Dense(
            self.total_arg_groups,
            activation="sigmoid",
            name="arg_group_output"
        )(wordvectors_nn_4)

        # arg classes (antibiotics)
        arg_class_output = keras.layers.Dense(
            self.total_arg_classes,
            activation="sigmoid",
            name="arg_class_output"
        )(wordvectors_nn_4)

        # Topology of the model
        _model = keras.models.Model(
            inputs=[
                wordvectors_input,
                # alignments_input
            ],
            outputs=[
                arg_class_output,
                arg_groups_output
            ]
        )

        return _model
