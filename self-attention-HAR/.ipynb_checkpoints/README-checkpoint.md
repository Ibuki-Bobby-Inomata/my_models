# Self-Attention-HAR
This is a self-attention based model to recognize human activitis with sensor.

Souce  [Human Activity Recognition from Wearable Sensor Data Using Self-Attention](https://github.com/saif-mahmud/self-attention-HAR)

## Sample
#### Setup model
```
from model.run_model import create_model

n_timesteps, n_features, n_outputs = train_val.shape[1], train_val.shape[2], output_shape

model = create_model(n_timesteps, n_features, n_outputs,d_model=128, nh=4, dropout_rate=0.2)
model.compile(loss='sparse_categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=4,verbose=1,min_delta=1e-4,mode='min')
```

#### run model
```
model.fit(train_val, train_label,
          epochs=100,
          batch_size=64,
          verbose=1,
          validation_split=0.1,
          callbacks=[reduce_lr_loss, earlyStopping])
```


### Architecture
![Architecture](/img/self-attention_arch.png) 