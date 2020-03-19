import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, concatenate, Dense, Conv1D, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np
from keras.layers.normalization import BatchNormalization


# 入力を定義
input1 = Input(shape=(30,4))
input2 = Input(shape=(30,8))

# 入力1から結合前まで(Conv1D)
acc_gy = tf.keras.layers.Conv1D(filters=64, kernel_size=10, dilation_rate=1, activation='relu')(input1)
acc_gy = tf.keras.layers.Conv1D(filters=64, kernel_size=10, dilation_rate=2, activation='relu')(acc_gy)

acc_gy = Model(inputs=input1, outputs=acc_gy)

# 入力2から結合前まで(Conv1D)
beacon = tf.keras.layers.Conv1D(filters=64, kernel_size=10, dilation_rate=1, activation='relu')(input2)
beacon = tf.keras.layers.Conv1D(filters=64, kernel_size=10, dilation_rate=2, activation='relu')(beacon)
beacon = Model(inputs=input2, outputs=beacon)

# 結合
combined = concatenate([acc_gy.output, beacon.output])
# combined = tf.keras.layers.GlobalMaxPooling1D()(combined)

# # 密結合
com_model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10, return_sequences=True))(combined)
com_model = tf.keras.layers.LSTM(units=n_hidden, batch_input_shape=(None, sequence_len, in_out_neurons), return_sequences=True)(com_model)

com_model = tf.keras.layers.LSTM(units=n_hidden, batch_input_shape=(None, sequence_len, in_out_neurons), return_sequences=False)(com_model)

com_model = tf.keras.layers.Dense(units=50, activation='relu')(com_model)
com_model = tf.keras.layers.Dense(class_num, activation='sigmoid')(com_model)

# モデル定義とコンパイル
model = Model(inputs=[acc_gy.input, beacon.input], outputs=com_model)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()