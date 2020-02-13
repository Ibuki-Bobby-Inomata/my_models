import tensorflow as tf
import keras


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, a, n):
        super(ResidualBlock, self).__init__()
        self.conv2d1x1_0 = tf.keras.layers.Conv2D(filters=a * (2 ** n), kernel_size=(1, 1), activation='relu')
        self.conv2d3x3 = tf.keras.layers.Conv2D(filters=a * (2 ** n), kernel_size=(3, 3), activation='relu')
        self.conv2d1x1_1 = tf.keras.layers.Conv2D(filters=(a ** 2) * (2 ** n), kernel_size=(1, 1), activation='relu')

    def call(self, input_tensor):
        x = self.conv2d1x1_0(input_tensor)
        x = self.conv2d3x3(x)
        x = self.conv2d1x1_1(x)

        return x, input_tensor


class ResidualBlocks_3(tf.keras.layers.Layer):
    def __init__(self, a, n):
        super(ResidualBlocks_3, self).__init__()
        self.reduce = tf.math.reduce_sum
        self.res_blocks = {}
        for i in range(1, 3):
            self.res_blocks[i] = ResidualBlock(a, n)

    def call(self, input_tensor):
        connections = []
        for i in range(1, 3):
            x, connection = self.res_blocks[i](input_tensor)
            connections.append(connection)
        x = self.reduce(connections, axis=0)

        return x


class ResidualBlocks_4(tf.keras.layers.Layer):
    def __init__(self, a, n):
        super(ResidualBlocks_4, self).__init__()
        self.reduce = tf.math.reduce_sum
        self.res_blocks = {}
        for i in range(1, 4):
            self.res_blocks[i] = ResidualBlock(a, n)

    def call(self, input_tensor):
        connections = []
        for i in range(1, 4):
            x, connection = self.res_blocks[i](input_tensor)
            connections.append(connection)
        x = self.reduce(connections, axis=0)

        return x


class ResidualBlocks_6(tf.keras.layers.Layer):
    def __init__(self, a, n):
        super(ResidualBlocks_6, self).__init__()
        self.reduce = tf.math.reduce_sum
        self.res_blocks = {}
        for i in range(1, 6):
            self.res_blocks[i] = ResidualBlock(a, n)

    def call(self, input_tensor):
        connections = []
        for i in range(1, 6):
            x, connection = self.res_blocks[i](input_tensor)
            connections.append(connection)
        x = self.reduce(connections, axis=0)

        return x

kernel_size = (7, 7)
input_shape = (35, 35, 1)
a = 4


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, input_shape=input_shape, activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2, 2))

model.add(ResidualBlocks_3(a=a, n=0))
model.add(ResidualBlocks_4(a=a, n=1))
model.add(ResidualBlocks_6(a=a, n=2))
model.add(ResidualBlocks_3(a=a, n=3))

model.add(tf.keras.layers.AveragePooling2D(2,2))

model.add(tf.keras.layers.GlobalMaxPooling2D())


# model.add(tf.keras.layers.Dense(units=10, activation='relu'))

model.add(tf.keras.layers.Dense(units=4, activation='softmax'))


model.summary()