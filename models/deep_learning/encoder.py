import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor
import tensorflow_addons as tfa

class ENCODERRegressor(DLRegressor):

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=1500,
            batch_size=16,
            loss="mean_squared_error",
            metrics=None
    ):



        self.name = "encoder"
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

        # conv block -1
        conv1 = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = tf.keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = tf.keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = tf.keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = tf.keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = tf.keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = tf.keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = tf.keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = tf.keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = tf.keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
        attention_softmax = tf.keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
        # attention mechanism
        attention_softmax = tf.keras.layers.Softmax()(attention_softmax)
        multiply_layer = tf.keras.layers.Multiply()([attention_softmax,attention_data])
        # last layer
        dense_layer = tf.keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = tf.keras.layers.Flatten()(dense_layer)
        output_layer = tf.keras.layers.Dense(units=1,activation='linear')(flatten_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(0.00001),
                      metrics=self.metrics)



        return model