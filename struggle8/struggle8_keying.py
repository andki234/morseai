# -*- coding: utf-8 -*-
 
import struggle8_base as s8b
import numpy as np
import keyboard
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class MorsePredictor:
    def __init__(self, filename):
        # define vecor for new training data
        self.training_vectors_data = []
        self.training_vectors_char = []

        # Set default training character
        self.training_char = 'A'

        # Load the trained model
        self.model = load_model(filename)

    # Function to decode Morse code sequences
    def decode_morse_sequence(self, morse_sequence,sequence_length=550):
        # Convert morse_sequence = "00010101" into numeric_sequence = [0, 0, 0, 1, 0, 1, 0, 1]
        morse_sequence = morse_sequence.strip('0') # remove all 0 from start and end of morse_sequence
        numeric_sequence = [int(char) for char in morse_sequence] 
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([numeric_sequence], maxlen=sequence_length, padding='post')
        prediction = self.model.predict(np.array(padded_sequence).reshape(1, -1, 1), verbose=0)
        predicted_index = np.argmax(prediction)
                
        # Find the corresponding Morse code symbol for the predicted index
        predicted_symbol = [symbol for symbol, index in s8b.unique_identifiers.items() if index == predicted_index][0]
        predicted_symbol

        return predicted_symbol

    
    def record_key_press(self, e): 
        if e.event_type == keyboard.KEY_DOWN:
            if e.name != 'space':  # Check if  the key pressed is not the spacebar
                self.is_key_down = True
        elif e.event_type == keyboard.KEY_UP: 
            if e.name != 'space':  # Check if the key released is not the spacebar
                self.is_key_down = False
        return False # Stop the key event from propagating
    
    # Collect training data from user
    def collect_training_data(self, binstr, char):
        # Ask user if they want to save the data
        save = input("Save data? (y/n): ")
        if save == 'y':
            self.training_vectors_data += [binstr]
            self.training_vectors_char += [char]
            self.combined_vectors = (self.training_vectors_char, self.training_vectors_data)
            # Save the training data to a file
            joblib.dump(self.combined_vectors, f"struggle4_retraining_data_for_char_{char}.pkl", protocol=0)
            print("Data saved.")


    def keying(self, save_data=False):
        while True:
            # Define the sampling rate in Hz
            sampling_rate = 300                

            # Create a flag to qtrack the current state of the keyboard
            self.is_key_down = False

            # Create an empty string to store the key presses
            key_presses = ""

            # Start recording key presses
            keyboard.hook(self.record_key_press)

            print("Press space when char has been keyed...")

            # Loop until ' ' key is pressed
            while True:
                if keyboard.is_pressed(' ') or keyboard.is_pressed('esc'):
                    break
                if self.is_key_down:
                    key_presses += "1"
                    print("1", end='', flush=True)  # Print without a new line
                else:
                    key_presses += "0"
                    print("0", end='', flush=True)  # Print without a new line
                time.sleep(1 / sampling_rate)

            # Stop recording key presses
            keyboard.unhook_all()

            key_presses = key_presses.strip('0') # remove all 0 from start and end of morse_sequence

            if len(key_presses) > 10:
                # predict text from morse code
                predicted_text = self.decode_morse_sequence(key_presses)
                print(f"\n\n\nPredicted Text: {predicted_text}")
                print(f"\nKey presses: {key_presses}")  

                # Collect training data
                if save_data:
                    self.collect_training_data(key_presses, self.training_char )

            while True:
                event = keyboard.read_event()       
                if event.event_type == (keyboard.KEY_UP or keyboard.KEY_DOWN):
                    if event.name == 'esc':
                        exit(0)
                    break

    # Menu for training system
    def training_menu(self):
        while True:
            print("\nTraing Menu:")
            print(f"Current training character: {self.training_char}")
            print("1: Set new training character")
            print("2: Start keying and store data")
            print("3: Test freely with trained model")
            print("0: Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                self.training_char = input("Enter the new training character: ")
            elif choice == '2':
                self.keying(save_data=True)
            elif choice == '3':
                self.keying(save_data=False)
            elif choice == '0':
                exit(0)
            else:
                print("Invalid choice. Try again.")

def main():
    mdecoder = MorsePredictor("model_epoch_12_loss_0.38.keras")
    #mdecoder.decode_morse_sequence("00000000001111110000001111111111111111110000000000")
    mdecoder.training_menu()
    
if __name__ == '__main__':
    main()
  
