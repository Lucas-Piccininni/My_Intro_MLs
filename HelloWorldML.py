import tensorflow as tf

# Create a simple model.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# Compile the model.
model.compile(loss='mean_squared_error', optimizer='sgd')

# Generate some synthetic data for training.
xs = [-1, 0, 1, 2, 3, 4]
ys = [-3, -1, 1, 3, 5, 7]

# Train the model.
model.fit(xs, ys, epochs=250)

# Make predictions using the trained model.
predictions = model.predict([5, 6, 7])
print(predictions)
