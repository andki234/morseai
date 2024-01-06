import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import random
import pickle
import math

class S7TClass():
    def __init__(self):
        # Morse code dictionary for training
        self.morse_code_dict = {
                'A': '.-',   'B': '-...', 'C': '-.-.', 'D': '-..',  'E': '.',
                'F': '..-.', 'G': '--.',  'H': '....', 'I': '..',   'J': '.---',
                'K': '-.-',  'L': '.-..', 'M': '--',   'N': '-.',   'O': '---',
                'P': '.--.', 'Q': '--.-', 'R': '.-.',  'S': '...',  'T': '-',
                'U': '..-',  'V': '...-', 'W': '.--',  'X': '-..-', 'Y': '-.--',
                'Z': '--..',

                '0': '-----', '1': '.----', '2': '..---', '3': '...--',
                '4': '....-', '5': '.....', '6': '-....', '7': '--...',
                '8': '---..', '9': '----.',

                '.': '.-.-.-', ',': '--..--', '?': '..--..', '\'': '.----.',
                '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
                '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
                '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.',
                '$': '...-..-', '@': '.--.-.',

                # Scandinavian characters
                'Å': '.--.-', 'Ä': '.-.-', 'Ö': '---.',

                # Additional international characters
                'É': '..-..', 'Ñ': '--.--', 'Ü': '..--'
            }
        
        # Create a unique numerical identifier for each Morse code symbol
        self.unique_identifiers = {symbol: i for i, symbol in enumerate(self.morse_code_dict.keys())}
        self.num_categories = len(self.unique_identifiers)


    def step_decay(self, epoch):
        initial_lr = self.initial_learning_rate # Initial learning rate
        drop = 0.75  # Factor by which the learning rate will be reduced
        epochs_drop = 20.0  # Every 'epochs_drop' epochs, the learning rate is reduced
        lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lr

    def create_model(self, lstm_units=64, lstm_final_units = 32, nof_lstm_layers = 1, dropout_rate = 0.33, learning_rate=0.001):
        # Model parameters
        self.initial_learning_rate=learning_rate

        # Create LSTM model with Dropout and BatchNormalization layers
        model = Sequential()
        for i in range(nof_lstm_layers):
            model.add(LSTM(lstm_units, input_shape=(None, 1), return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_final_units))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.num_categories, activation='softmax'))

        # Compile the model
        optimizer = Adam(learning_rate=self.initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model
    
    def train_model(self, model, X, y, epochs=20, batch_size=64):
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        lr_scheduler = LearningRateScheduler(self.step_decay)

        model_checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',   # Monitor validation loss
                    filepath = "model_epoch_{epoch:02d}_loss_{loss:.2f}.keras",        # Change the formatting as needed
                    save_best_only=True,                                            # Save only the best model                                        
                    mode='auto',                                                     # Automatically select the mode: min or max
                    verbose=1,                                                       # Print the model checkpointing progress
                )

        # Train the model
        history = model.fit(np.array(X).reshape(len(X), -1, 1), 
                            np.array(y), validation_split=0.2, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            callbacks=[model_checkpoint_callback, early_stopping, lr_scheduler],)

        # Save history with pickle (only the history attribute)
        with open('struggle7_model_history.pkl', 'wb') as f:  # Use .pkl as the file extension for clarity
            pickle.dump(history.history, f)

        save_model(model, 'struggle8_model.keras')
        save_model(model, 'struggle8_model.h5')

        
if __name__ == '__main__':
    s7tc = S7TClass()
    model = s7tc.create_model(lstm_units=196, lstm_final_units = 64, nof_lstm_layers = 2, dropout_rate = 0.2, learning_rate=0.0005)
    # Load the model with pickle
    with open('struggle8_training_data.pkl', 'rb') as f:  # Use .pkl as the file extension for clarity
        X, y = pickle.load(f)
   
    print("Shape of y: ", np.array(y).shape)
    print("Shape of X: ", np.array(X).reshape(len(X), -1, 1).shape)
    s7tc.train_model(model, X, y, epochs=50, batch_size=64)

    print("Done!")


