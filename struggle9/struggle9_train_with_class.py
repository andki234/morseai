import struggle9_base as base
import tensorflow as tf
import numpy as np
import random
import pickle
import math
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

class S7TClass():
    def __init__(self):
        # Morse code dictionary for training
        self.morse_code_dict = base.morse_code_dict
                
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
    
    def train_model2(self, model, X, y, epochs=20, batch_size=64):
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

        save_model(model, 'struggle9_model.keras')
        save_model(model, 'struggle9_model.h5')
        
    def train_model(self, model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):

        #Split the data into training and temporary sets (50% each)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        print("Shape of y_train_temp: ", np.array(y_train).shape)
        print("Shape of X_train_temp: ", np.array(X_train).reshape(len(X_train), -1, 1).shape)
        print("Shape of y_test: ", np.array(y_test).shape)
        print("Shape of X_test: ", np.array(X_test).reshape(len(X_test), -1, 1).shape)
        print("Batch size: ", batch_size)

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
        lr_scheduler = LearningRateScheduler(self.step_decay)

        model_checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',   # Monitor validation loss
                    filepath = "model_epoch_{epoch:02d}_loss_{loss:.2f}.keras",        # Change the formatting as needed
                    save_best_only=True,                                            # Save only the best model
                    mode='auto',                                                     # Automatically select the mode: min or max
                    verbose=1,                                                       # Print the model checkpointing progress
                )

        # Train the model
        history = model.fit(np.array(X_train).reshape(len(X_train), -1, 1),
                            np.array(y_train),
                            validation_data=(np.array(X_test).reshape(len(X_test), -1, 1),
                                            np.array(y_test)),
                            #validation_split=0.2,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint_callback, early_stopping, lr_scheduler])

        # Save history with pickle (only the history attribute)
        with open('struggle7_model_history.pkl', 'wb') as f:  # Use .pkl as the file extension for clarity
            pickle.dump(history.history, f)

        save_model(model, 'struggle8_model.keras')

        
if __name__ == '__main__':
    s7tc = S7TClass()
    model = s7tc.create_model(lstm_units=64, lstm_final_units =64, nof_lstm_layers =4, dropout_rate = 0.2, learning_rate=0.001)
    # Load the training data
    with open('struggle9_training_data.pkl', 'rb') as f:  # Use .pkl as the file extension for clarity
        X_train, y_train = pickle.load(f)
    # Load the test data
    with open('struggle9_test_data.pkl', 'rb') as f:  # Use .pkl as the file extension for clarity
        X_test, y_test = pickle.load(f)

    # Train the model
    s7tc.train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Evaluate the model
    print("Evaluating model...")
    print("Shape of y_test: ", np.array(y_test).shape)
    print("Shape of X_test: ", np.array(X_test).reshape(len(X_test), -1, 1).shape)
    
    score = model.evaluate(np.array(X_test).reshape(len(X_test), -1, 1), np.array(y_test), verbose=1)
    print('Test loss:', score[0])
    print('Test loss in %:', score[0] * 100 / len(y_test))
    print('Test accuracy:', score[1])
    print('Test accuracy in %:', score[1] * 100 / len(y_test))
    
    print("----")
    
    score = model.evaluate(np.array(X_train).reshape(len(X_train), -1, 1), np.array(y_train), verbose=1)
    print('Train loss:', score[0])
    print('Train loss in %:', score[0] * 100 / len(y_train))
    print('Train accuracy:', score[1])
    print('Train accuracy in %:', score[1] * 100 / len(y_train))
    
    
  
    print("Done!")

