import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras import layers, models
from keras.datasets import imdb


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def vectorize_sequences(sequences, dimension=5000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


num_words = 5000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
X_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')
X_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(num_words,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', recall, precision])

x_val = X_train[:num_words]
partial_x_train = X_train[num_words:]
y_val = y_train[:num_words]
partial_y_train = y_train[num_words:]
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
epochs = range(1, len(history_dict['acc'])+1)

plt.plot(epochs, history_dict['recall'], 'bo', label='recall')
plt.plot(epochs, history_dict['val_recall'], 'b', label='val_recall')
plt.xlabel('epoch')
plt.ylabel('recall')
plt.legend()
plt.savefig("recall.png")

plt.plot(epochs, history_dict['precision'], 'bo', label='precision')
plt.plot(epochs, history_dict['val_precision'], 'b', label='val_precision')
plt.xlabel('epoch')
plt.ylabel('precision')
plt.legend()
plt.savefig("precision.png")
plt.clf()

plt.plot(epochs, history_dict['acc'], 'bo', label='accuracy')
plt.plot(epochs, history_dict['val_acc'], 'b', label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig("accuracy.png")

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(num_words,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=4, batch_size=512)
model.save('imdb.h5')
print(model.evaluate(X_test, y_test))
