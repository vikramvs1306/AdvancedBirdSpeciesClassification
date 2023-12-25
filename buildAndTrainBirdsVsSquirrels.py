import tensorflow as tf
import preprocessDefinition as preprocess
from matplotlib import pyplot as plt
import pandas as pd

def preporc(raw_dataset):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    dataset = raw_dataset.map(lambda serialized_examples: preprocess.parse_examples(serialized_examples, feature_description), num_parallel_calls=16).map(
        lambda data,label: (tf.keras.applications.xception.preprocess_input(data), label), num_parallel_calls=16
        ).batch(32)
    return dataset

train_raw_dataset = tf.data.TFRecordDataset(["./birds-vs-squirrels-train.tfrecords"])
valid_raw_dataset = tf.data.TFRecordDataset(["./birds-vs-squirrels-validation.tfrecords"])

# you can edit batch size and num_parallel calls below based on your architecture
train_dataset = preporc(train_raw_dataset)
valid_dataset = preporc(valid_raw_dataset)

# Model Creation
base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(3, activation="softmax")(avg)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
model.summary()

for layer in base_model.layers:
    layer.trainable = False

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('birdsVsSquirrelsModel', save_best_only=True)
earlyStop_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
ss=5e-1
optimizer = tf.keras.optimizers.SGD(learning_rate=ss)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_dataset, validation_data=valid_dataset, epochs=25, callbacks=[checkpoint_cb, earlyStop_cb])

model.save('birdsVsSquirrelsModel')

# Plotting the accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

############################*********************############################
# Unfreeze layers train the lower layers
model = tf.keras.models.load_model('birdsVsSquirrelsModel')

for layer in base_model.layers:
    layer.trainable = True

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('birdsVsSquirrelsModel', save_best_only=True)
earlyStop_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
ss=3e-2
optimizer = tf.keras.optimizers.SGD(learning_rate=ss)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(train_dataset, validation_data=valid_dataset, epochs=10, callbacks=[checkpoint_cb, earlyStop_cb])

# Plotting the accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()