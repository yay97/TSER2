# MLP model
import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor




class MLPRegressor(DLRegressor):

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=2000,
            batch_size=16,
            loss="mean_squared_error",
            metrics=None
    ):



        self.name = "MLP"
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
        input_layer = tf.keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = tf.keras.layers.Flatten()(input_layer)

        layer_1 = tf.keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = tf.keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = tf.keras.layers.Dropout(0.2)(layer_1)
        layer_2 = tf.keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = tf.keras.layers.Dropout(0.2)(layer_2)
        layer_3 = tf.keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = tf.keras.layers.Dropout(0.3)(layer_3)
        output_layer = tf.keras.layers.Dense(1, activation='linear')(output_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adadelta(),
                      metrics=self.metrics)




        return model