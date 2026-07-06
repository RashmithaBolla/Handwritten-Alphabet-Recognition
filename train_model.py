import tensorflow as tf
import tensorflow_datasets as tfds

# Load EMNIST letters dataset
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# ✅ Fixed preprocessing - NO inversion needed
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    
    # EMNIST is already white text on black background
    # Keep as-is (0=black background, 255=white text)
    image = tf.reshape(image, (28, 28, 1))
    
    # Labels are 1–26 → convert to 0–25
    label = label - 1
    
    return image, label

# ✅ Add data augmentation to handle different writing styles
def augment(image, label):
    # Random slight rotation (±10 degrees)
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
    
    # Random small shifts
    image = tf.image.random_crop(image, size=(24, 24, 1))
    image = tf.image.resize(image, (28, 28))
    
    # Random slight brightness adjustment
    image = image + tf.random.uniform([], -0.1, 0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess).cache()
ds_train = ds_train.map(augment).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess).batch(64).cache()

# ✅ Improved model with regularization
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

model.fit(ds_train, epochs=5, validation_data=ds_test, callbacks=callbacks)

# Save model
model.save("emnist_cnn_fixed.keras")