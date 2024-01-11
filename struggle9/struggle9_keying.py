# -*- coding: utf-8 -*-
 
import struggle9_base as base
import numpy as np
import keyboard
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class MorsePredictor:
    def __init__(self):
        # define vecor for new training data
        self.training_vectors_data = []
        self.training_vectors_char = []

        # Set default training character
        self.training_char = 'A'

        # Load the trained model
        self.model =load_model('struggle9_model.keras')

    # Function to decode Morse code sequences
    def decode_morse_sequence(morse_sequence, takt, model, unique_identifiers):
        decoded_sequence = ''
        for morse_char in morse_sequence.split(' '):
            if morse_char:  # Avoid empty strings
                numeric_sequence = base.morse_to_timestep_numeric(morse_char, base.generate_timing(swe_takt=takt))
                #print(f"numeric_sequence: {numeric_sequence}")
                padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([numeric_sequence], maxlen=462, padding='post')
                prediction = model.predict(np.array(padded_sequence).reshape(1, -1, 1), verbose=0)
                predicted_index = np.argmax(prediction)
                
                # Find the corresponding Morse code symbol for the predicted index
                predicted_symbol = [symbol for symbol, index in unique_identifiers.items() if index == predicted_index][0]
                decoded_sequence += predicted_symbol

        return decoded_sequence

    
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

            # remove all 0 from start and end of key_presses
            key_presses = key_presses.rstrip('0')
            key_presses = key_presses.lstrip('0')

            if len(key_presses) > 10:
                # predict text from morse code
                predicted_text = self.predict(key_presses)
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
            print("3: Test frely with trained model")
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
    mdecoder = MorsePredictor()
    mdecoder.training_menu()
    
if __name__ == '__main__':
    main()
  