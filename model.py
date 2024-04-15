# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:02:45 2023

@author: NENO
"""

#Para cargar y operar con el dataset
import pandas as pd
import numpy as np

#Para entrenar y crear el modelo de la red convolucional
import tensorflow as tf
from keras import layers, models, Input, optimizers, callbacks

#Para imprimir el modelo
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Define a normalization function
def normalize_column(matrix):
    norm = np.linalg.norm(matrix)
    if norm == 0:
        return matrix
    return matrix / norm

def normalize(matrix: np.ndarray):
    positive_elements = (matrix > 0)
    negative_elements = (matrix < 0)
    zero_elements = (matrix == 0)
    augmented_array = np.zeros_like(matrix)
    augmented_array[positive_elements] = 1
    augmented_array[negative_elements] = -1
    augmented_array[zero_elements] = 0
    return augmented_array

df = pd.read_json('Dataset/MatrizPosiciones.json')

# Specify the percentage for the test set (30% in this case)
test_size = 0.2 # Changed to 20% with 10% validation

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_df_copy = train_df.copy()
test_df_copy = test_df.copy()
X_train = tf.constant(np.array(train_df.drop("y",axis=1).values.tolist()).transpose(0, 2, 3, 1))
y_train = tf.constant(train_df.drop(["posiciones", "pawns", "knights","bishops", "rooks", "queens", "kings"],axis=1).values)


X_test = tf.constant(np.array(test_df.drop("y",axis=1).values.tolist()).transpose(0, 2, 3, 1))
y_test = tf.constant(test_df.drop(["posiciones", "pawns", "knights","bishops", "rooks", "queens", "kings"],axis=1).values)


#Creamos el modelo de la red convolucional

model = models.Sequential()
model.add(Input(shape=(8, 8, 7)))
model.add(layers.Conv2D(128, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(256, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(256, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(256, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1))
model.summary()


#optimizer = tf.keras.optimizers.RMSprop(0.01)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#Entrenamos el modelo

history = model.fit(X_train, y_train, epochs=60, 
                    validation_split=0.1)


plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label = 'val_mae')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(loc='lower right')
plt.savefig("Ex1.png")
plt.clf()

metrics_df = pd.DataFrame(history.history)
metrics_df[["loss","val_loss"]].plot()
plt.savefig("Ex2.png")
plt.clf()

y_pred = model.predict(X_test)
orig_residuals = np.array(y_test) - y_pred
std_residuals = orig_residuals / np.std(orig_residuals)
print(f"RMSE: {(np.sum(orig_residuals ** 2) / len(orig_residuals)) ** (1/2)}")
plt.scatter(y_pred, std_residuals, color="red")
plt.axhline(0, linestyle="dashed")
plt.title("Std. Residuals vs Prediction")
plt.savefig("Ex5.png")
plt.clf()
plt.hist(std_residuals)
plt.savefig("Ex6.png")
plt.clf()

df = pd.read_json('Dataset/MatrizPosiciones2.json')

train_new_df, test_new_df = train_test_split(df, test_size=test_size, random_state=42)
train_new_df = train_new_df.reset_index(drop=True)
train_df_copy = pd.concat([train_df_copy, train_new_df])

X_train_copy = tf.constant(normalize(np.array(train_df_copy.drop(['posiciones', "y"],axis=1).values.tolist()).transpose(0, 2, 3, 1)))
y_train_copy = tf.constant(train_df_copy.drop(["posiciones", "pawns", "knights","bishops", "rooks", "queens", "kings"],axis=1).values)

X_test_copy = tf.constant(normalize(np.array(test_df_copy.drop(['posiciones', "y"],axis=1).values.tolist()).transpose(0, 2, 3, 1)))
y_test_copy = tf.constant(test_df_copy.drop(["posiciones", "pawns", "knights","bishops", "rooks", "queens", "kings"],axis=1).values)
model = models.Sequential()
model.add(Input(shape=(8, 8, 6)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1))
model.summary()

optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(X_train_copy, y_train_copy, epochs=45, 
                    validation_split=0.1)


plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label = 'val_mae')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig("Ex3.png")
plt.clf()

metrics_df = pd.DataFrame(history.history)
metrics_df[["loss","val_loss"]].plot()
plt.savefig("Ex4.png")
plt.clf()

y_pred = model.predict(X_test_copy)
residuals = np.array(y_test_copy) - y_pred
std_residuals = residuals / np.std(residuals)
print(f"RMSE: {(np.sum(residuals ** 2) / len(residuals)) ** (1/2)}")
plt.scatter(y_pred, std_residuals, color="red")
plt.axhline(0, linestyle="dashed")
plt.title("Std. Residuals vs Prediction")
plt.savefig("Ex7.png")
plt.clf()
plt.hist(std_residuals)
plt.savefig("Ex8.png")