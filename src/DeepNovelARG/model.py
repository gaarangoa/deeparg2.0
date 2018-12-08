import tensorflow as tf
from tensorflow import keras


class DeepARG():
    def __init__(self, input_dataset_wordvectors_size=0, input_convolutional_dataset_size=2000, num_classes=[], num_groups=[]):
        # setup parameters for model
        self.input_dataset_wordvectors_size = input_dataset_wordvectors_size
        self.input_convolutional_dataset_size = input_convolutional_dataset_size
        self.total_arg_classes = num_classes
        self.total_arg_groups = num_groups

    def model(self):

        ################################################
        ################## Convolution #################
        ################################################
        convolutional_input = keras.Input(
            shape=(self.input_convolutional_dataset_size, 1),
            name="convolutional_input"
        )

        conv_nn = keras.layers.Conv1D(
            16, 3,
            input_shape=(None, self.input_convolutional_dataset_size, 1),
            activation='elu',
            padding='same',
            name='encoder_conv0',
            kernel_initializer='he_uniform'
        )(convolutional_input)
        conv_nn = keras.layers.BatchNormalization(name='encoder_bn0')(conv_nn)
        conv_nn = keras.layers.Conv1D(
            24, 3,
            activation='elu',
            padding='same',
            name='encoder_conv1',
            kernel_initializer='he_uniform'
        )(conv_nn)
        conv_nn = keras.layers.BatchNormalization(name='encoder_bn1')(conv_nn)
        conv_nn = keras.layers.MaxPooling1D()(conv_nn)

        conv_nn = keras.layers.Conv1D(
            32, 5,
            activation='elu',
            padding='same',
            name='encoder_conv2',
            kernel_initializer='he_uniform'
        )(conv_nn)
        conv_nn = keras.layers.BatchNormalization(name='encoder_bn2')(conv_nn)
        conv_nn = keras.layers.Conv1D(
            48, 5,
            activation='elu',
            padding='same',
            name='encoder_conv3',
            kernel_initializer='he_uniform'
        )(conv_nn)
        conv_nn = keras.layers.BatchNormalization(name='encoder_bn3')(conv_nn)
        conv_nn = keras.layers.MaxPooling1D()(conv_nn)

        conv_nn = keras.layers.Conv1D(
            64, 7,
            activation='elu',
            padding='same',
            name='encoder_conv4',
            kernel_initializer='he_uniform'
        )(conv_nn)
        conv_nn = keras.layers.BatchNormalization(name='encoder_bn4')(conv_nn)
        conv_nn = keras.layers.Conv1D(
            96, 7,
            activation='elu',
            padding='same',
            name='encoder_conv5',
            kernel_initializer='he_uniform'
        )(conv_nn)
        conv_nn = keras.layers.BatchNormalization(name='encoder_bn5')(conv_nn)
        conv_nn = keras.layers.MaxPooling1D()(conv_nn)

        conv_nn = keras.layers.Flatten()(conv_nn)

        ################################################
        ################### Word Vectors ###############
        ################################################
        # Input layer
        wordvectors_input = keras.Input(
            shape=(self.input_dataset_wordvectors_size,),
            name="wordvectors_input"
        )

        # Hiden Layer
        wv_nn = keras.layers.Dense(
            1000,
            activation='relu'
        )(wordvectors_input)
        wv_nn = keras.layers.Dropout(0.2)(wv_nn)
        wv_nn = keras.layers.Dense(
            800,
            activation='relu'
        )(wv_nn)
        wv_nn = keras.layers.Dropout(0.2)(wv_nn)
        wv_nn = keras.layers.Dense(
            400,
            activation='relu'
        )(wv_nn)
        wv_nn = keras.layers.Dropout(0.2)(wv_nn)
        wv_nn = keras.layers.Dense(
            200,
            activation='relu'
        )(wv_nn)

        ################################################
        #################### Merge #####################
        ################################################

        latent = keras.layers.concatenate(
            [wv_nn, conv_nn]
        )

        ################################################
        ################### Output #####################
        ################################################

        # arg groups (names)
        arg_groups_hidden_input = keras.layers.Dense(
            300,
            activation='relu'
        )(wv_nn)
        arg_groups_output = keras.layers.Dense(
            self.total_arg_groups,
            activation="softmax",
            name="arg_group_output"
        )(arg_groups_hidden_input)

        # arg classes (antibiotics)
        arg_class_hidden_input = keras.layers.Dense(
            100,
            activation='relu'
        )(latent)
        arg_class_output = keras.layers.Dense(
            self.total_arg_classes,
            activation="softmax",
            name="arg_class_output"
        )(arg_class_hidden_input)

        # Topology of the model
        _model = keras.models.Model(
            inputs=[
                wordvectors_input,
                convolutional_input
            ],
            outputs=[
                arg_class_output,
                arg_groups_output
            ]
        )

        return _model
