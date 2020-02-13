import tensorflow as tf
import keras


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n):
        super(ResidualBlock, self).__init__()
        self.conv2d3x3 = tf.keras.layers.Conv2D(filters=64 * (2 ** n), kernel_size=(3, 3))

    def call(self, input_tensor):
        x = self.conv2d3x3(input_tensor)
        x = self.conv2d3x3(x)

        return x, input_tensor


class ResidualBlocks_3(tf.keras.layers.Layer):
    def __init__(self, n):
        super(ResidualBlocks_3, self).__init__()
        self.reduce = tf.math.reduce_sum

        self.res_blocks = {}
        for i in range(1, 3):
            self.res_blocks[i] = ResidualBlock(n)

    def call(self, input_tensor):
        connections = []
        for i in range(1, 3):
            x, connection = self.res_blocks[i](input_tensor)
            connections.append(connection)
        x = self.reduce(connections, axis=0)

        return x


class ResidualBlocks_4(tf.keras.layers.Layer):
    def __init__(self, n):
        super(ResidualBlocks_4, self).__init__()
        self.res_blocks = {}
        for i in range(1, 4):
            self.res_blocks[i] = ResidualBlock(n)

    def call(self, input_tensor):
        connections = []
        for i in range(1, 4):
            x, connection = self.res_blocks[i](input_tensor)
            connections.append(connection)
        x = self.reduce(connections, axis=0)

        return x


class ResidualBlocks_6(tf.keras.layers.Layer):
    def __init__(self, n):
        super(ResidualBlocks_6, self).__init__()
        self.res_blocks = {}
        for i in range(1, 6):
            self.res_blocks[i] = ResidualBlock(n)

    def call(self, input_tensor):
        connections = []
        for i in range(1, 6):
            x, connection = self.res_blocks[i](input_tensor)
            connections.append(connection)
        x = self.reduce(connections, axis=0)

        return x



kernel_size = (7, 7)
input_shape = (35, 35, 1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, input_shape=input_shape, activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2, 2))

model.add(ResidualBlocks_3(0))
model.add(ResidualBlocks_4(1))
model.add(ResidualBlocks_6(2))
model.add(ResidualBlocks_3(3))

model.add(tf.keras.layers.AveragePooling2D(2, 2))
model.add(tf.keras.layers.GlobalMaxPool2D())

model.add(tf.keras.layers.Dense(units=128, activation='relu'))

model.add(tf.keras.layers.Dense(units=4, activation='softmax'))


model.summary()