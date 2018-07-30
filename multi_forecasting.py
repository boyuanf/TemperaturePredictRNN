def generator_multi_targets(data, lookback, delay, min_index, max_index,
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
        targets = np.zeros((len(rows), lookback // step, 1))
        for j, row in enumerate(rows):
            indices_data = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices_data]
            indices_target = range(rows[j] + delay - lookback, rows[j] + delay, step)
            # print(j)
            # print(indices_target)
            # print(data[indices_target].shape)
            targets[j] = data[indices_target][:, 1].reshape(lookback // step, 1)
        yield samples, targets



---------------------------------

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_mul = generator_multi_targets(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
test_mul = generator_multi_targets(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    shuffle=False,
                    step=step,
                    batch_size=batch_size)

test_steps = (300000 - 200001 - lookback)


-------------------------------------------

samples, targets = next(test_mul)
print(samples.shape)
print(targets.shape)
-------------------------------------------

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.TimeDistributed(layers.Dense(1, input_dim=(lookback // step, 32))))

model.summary()

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_mul,
                              steps_per_epoch=500,
                              epochs=20,
                              #validation_data=val_gen,
                              #validation_steps=val_steps
                              )

-------------------------------------------------
predicts = model.predict_generator(test_mul, steps=100, workers=4)
------
predicts.shape
----------
test = predicts[:, 1, 0]
plt.plot(range(1200), test[:1200])
-----------
adjoint_predicts = []
i = 0
while i * predicts.shape[1] * 6 < predicts.shape[0]:
    adjoint_predicts.append(predicts[i * predicts.shape[1] * 6, :])
    i = i + 1

adjoint_predicts_array = np.vstack(adjoint_predicts)
adjoint_predicts_array = adjoint_predicts_array.reshape(-1)
----------
adjoint_predicts_array.shape
-----------------------
plt.plot(range(600), adjoint_predicts_array[:600])
-----------------------
targets = []
for step in range(100):
    sample_batch, target_batch = next(test_mul)
    targets.append(target_batch)

targets = np.vstack(targets)
----------------------
targets.shape
--------------------
import matplotlib.pyplot as plt

epochs = range(0, 240)

plt.figure()
plt.plot(epochs, targets[0,:,0], 'r', label='0')
plt.plot(epochs, targets[6*240,:,0], 'b', label='50+')
plt.title('Predicts and targets compare')
plt.legend()

plt.show()
------------------
adjoint_targets = []
i = 0
while i * targets.shape[1] * 6 < targets.shape[0]:
    adjoint_targets.append(targets[i * targets.shape[1] * 6, :])
    i = i + 1

adjoint_targets_array = np.vstack(adjoint_targets)
adjoint_targets_array = adjoint_targets_array.reshape(-1)
plt.plot(range(600), adjoint_targets_array[:600])
-------------------
import matplotlib.pyplot as plt

epochs = range(0, 200)

plt.figure()
plt.plot(epochs, adjoint_targets_array[:200], 'r', label='targets')
plt.plot(epochs, adjoint_predicts_array[:200], 'b', label='predicts')
plt.title('Predicts and targets compare')
plt.legend()

plt.show()
--------------------------


model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64,
                     activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(1, input_dim=(lookback // step, 64))))

model.summary()

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_mul,
                              steps_per_epoch=500,
                              epochs=20,
                              #validation_data=val_gen,
                              #validation_steps=val_steps
                              )

-----------------------------------------