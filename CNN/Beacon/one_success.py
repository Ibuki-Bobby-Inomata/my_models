import tensorflow as tf
keras = tf.keras


class Conv2DDila(tf.keras.layers.Layer):
    def __init__(self):
        super(Conv2DDila, self).__init__()
        self.inputly = tf.keras.layers.Conv2D(64, (5, 1), dilation_rate=1, input_shape=(35, 1, 1), activation="relu")

        self.middle_1 = tf.keras.layers.Conv2D(64, (5, 1), dilation_rate=1, activation="relu")
        self.middle_2 = tf.keras.layers.Conv2D(64, (5, 1), dilation_rate=2, activation="relu")
        self.middle_3 = tf.keras.layers.Conv2D(64, (5, 1), dilation_rate=3, activation="relu")

        self.drop = tf.keras.layers.Dropout(0.5)

        self.maxpooling = tf.keras.layers.GlobalMaxPooling2D()

    def call(self, input_tensor):
        x = self.inputly(input_tensor)
        x = self.drop(x)
        x = self.middle_1(x)
        x = self.drop(x)
        x = self.middle_2(x)
        x = self.drop(x)
        x = self.middle_3(x)
        x = self.drop(x)
        return self.maxpooling(x)

class OneSucModule(tf.keras.Model):
    def __init__(self):
        super(OneSucModule, self).__init__()
        self.conv2d = Conv2DDila()
        self.dense_200 = tf.keras.layers.Dense(units=200, activation="relu")
        self.dense_100 = tf.keras.layers.Dense(units=100, activation="relu")

        self.last_dense = tf.keras.layers.Dense(4)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.conv2d(x)
        x = self.dense_200(x)
        x = self.dense_100(x)

        return self.softmax(x)