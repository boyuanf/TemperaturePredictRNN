import os
import numpy as np
from keras import layers
from keras.models import Model
from matplotlib import pyplot as plt


data_dir = '/home/ubuntu/Boyuan/jena_climate_2009_2016'
# data_dir = 'C:\Boyuan\Machine Learning\Datasets\jena_climate_2009_2016'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

# Read the data
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# Normalize the dataset
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

# Create the training, validation, test dataset

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_sqn = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_sqn = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    shuffle=False,
                    step=step,
                    batch_size=batch_size)
test_sqn = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300001,
                    max_index=None,
                    shuffle=False,
                    step=step,
                    batch_size=batch_size)

#val_steps = (300000 - 200001 - lookback)
val_steps = 500

test_steps = (len(float_data) - 300001 - lookback)

# Train the model

inputs = layers.Input(shape=(None, float_data.shape[-1]))
gru = layers.GRU(32, dropout=0.5, recurrent_dropout=0.5)(inputs)
outputs = layers.Dense(1)(gru)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop',
              loss='mae',
              metrics=['mae'])
history = model.fit_generator(train_sqn,
                    steps_per_epoch=500,
                    epochs=40,)
                    #validation_data=val_sqn,
                    #validation_steps=val_steps)


# Save training and validation result

loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('Training and validation loss.png')


# Evaluate the training model
scoreSeg = model.evaluate_generator(test_sqn, steps=1000, workers=4)
print("mean_absolute_error = ", scoreSeg)

# Compare predicts and targets
test_sqn = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300001,
                    max_index=None,
                    shuffle=False,
                    step=step,
                    batch_size=batch_size)

predicts = model.predict_generator(test_sqn, steps=100, workers=4)

test_sqn = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300001,
                    max_index=None,
                    shuffle=False,
                    step=step,
                    batch_size=batch_size)

targets = []
for step in range(100):
    sample_batch, target_batch = next(test_sqn)
    targets.append(target_batch)

targets = np.hstack(targets)

epochs = range(0, 1200)

plt.figure()
plt.plot(epochs, predicts[:1200], 'r', label='predicts')
plt.plot(epochs, targets[:1200], 'b', label='targets')
plt.title('Predicts and targets compare')
plt.legend()

plt.savefig('Predicts and targets compare.png')





