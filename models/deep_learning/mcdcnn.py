import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor


class MCDCNNRegressor(DLRegressor):

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=10,
            batch_size=16,
            loss="mean_squared_error",
            metrics=None
    ):

        self.name = "mcdcnn"
        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics
        )


    def build_model(self, input_shape):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'

        if n_t < 60: # for ItalyPowerOndemand
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = tf.keras.layers.Input((n_t,1))
            input_layers.append(input_layer)

            conv1_layer = tf.keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(input_layer)
            conv1_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = tf.keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(conv1_layer)
            conv2_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = tf.keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = tf.keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = tf.keras.layers.Dense(units=732,activation='relu')(concat_layer)

        output_layer = tf.keras.layers.Dense(1, activation='linear')(fully_connected)

        model = tf.keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=0.0005),
                      metrics=self.metrics)



        return model