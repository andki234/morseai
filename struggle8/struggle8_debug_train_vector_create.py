import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
import datetime
import os
from itertools import product
import math
import pickle

class MorseCodeTrainingDataGeneratorClass:
    def __init__(self, sample_rate=300):
        self.morse_code_dict = {
            'A': '.-',   'B': '-...',
        }

        # Set sample rate
        self.sample_rate = sample_rate 

        # Create a unique numerical identifier for each Morse code symbol
        self.unique_identifiers = {symbol: i for i, symbol in enumerate(self.morse_code_dict.keys())}
        self.num_categories = len(self.unique_identifiers)


    # Count morse units for a character
    def count_morse_units(self, char):
        if char.lower() in self.morse_code_dict:
            morse_sequence = self.morse_code_dict[char.lower()]
            #print(f"morse_sequence: {morse_sequence} : {len(morse_sequence)}")
            #print(f"sum: {sum(1 if symbol == '.' else 3 for symbol in morse_sequence)}")
            return sum(1 if symbol == '.' else 3 for symbol in morse_sequence) + len(morse_sequence) - 1
        return 0

    # Count morse units in a string
    def count_morse_units_in_string(self, s):
        total_units = 0

        for char in s:
            # Count units for the current character
            total_units += self.count_morse_units(char)

            # Add 7 units for space
            if char.isspace():
                total_units += 7

        if (s[len(s) - 1].isspace()):
            total_units += 3 * (len(s) - 2)  # Add 3 units for space between characters, except for the last character if it is a space
        else:
            total_units += 3 * (len(s) - 1) # Add 3 units for space between characters

        return total_units
  
    def generate_timing(self, swe_takt=48):
        # Optimized generate_timing method
        # Calculate no of units in PARIS from dictionary when dot is 1, dash is 3, space between elements is 1, space between letters is 3 and space between words is 7
        # P = .--. 11
        # PAUS 3
        # A = .- 5
        # PAUS 3
        # R = .-. 7
        # PAUS 3
        # I = .. 3
        # PAUS 3
        # S = ... 5
        # PAUS 7
        # Total 11 + 3 + 5 + 3 + 7 + 3 + 3 + 3 + 5 + 7 = 50

        # Calculate no of units in PARIS from dictionary when dot is 1, dash is 3, space between elements is 1, space between letters is 3 and space between words is 7
        no_units_in_paris_inc_space_between_words = self.count_morse_units_in_string('PARIS ')

        # Ten PARIS is 500 units and must be sent in 60 seconds to get 10 WPM
        ten_units_in_paris_inc_space_between_words = no_units_in_paris_inc_space_between_words * 10

        # Calculate baud from ten PARIS (500 / 60 = 8.333) and modify for given swe_takt
        bd = (ten_units_in_paris_inc_space_between_words / 60.0) * (1 / 50 * swe_takt)

        # Calculate dot duration in seconds from bd (8.333) (1 / 8.333 = 0.12)
        dot_duration_in_seconds = 1.0 / bd

        # Calculate dash duration in seconds from bd (8.333) (3 / 8.333 = 0.36)
        dash_duration_in_seconds = 3.0 / bd

        # Calculate space between elements in seconds from bd (8.333) (1 / 8.333 = 0.12)
        space_between_elements_in_seconds = 1.0 / bd

        # Calculate space between characters in seconds from bd (8.333) (3 / 8.333 = 0.36)
        space_between_characters_in_seconds = 3.0 / bd

        # Calculate space between words in seconds from bd (8.333) (7 / 8.333 = 0.84)
        space_between_words_in_seconds = 7.0 / bd

        # Calculate samples per dot
        samples_per_dot = int(dot_duration_in_seconds * self.sample_rate)
        # Calculate samples per dash
        samples_per_dash = int(dash_duration_in_seconds * self.sample_rate)
        # Calculate samples per space between elements
        samples_per_space_between_elements = int(space_between_elements_in_seconds * self.sample_rate)
        # Calculate samples per space between characters
        samples_per_space_between_characters = int(space_between_characters_in_seconds * self.sample_rate)
        # Calculate samples per space between words
        samples_per_space_between_words = int(space_between_words_in_seconds * self.sample_rate)

        # Return dictionary with timing
        return {
            'dotduration': dot_duration_in_seconds,
            'dashduration': dash_duration_in_seconds,
            'space_between_elements_duration': space_between_elements_in_seconds,
            'space_between_characters_duration': space_between_characters_in_seconds,
            'space_between_words_duration': space_between_words_in_seconds,
            'dot': samples_per_dot,
            'dash': samples_per_dash,
            'intra_char_space': samples_per_space_between_elements,
            'inter_char_space': samples_per_space_between_characters,
            'inter_word_space': samples_per_space_between_words,
            'sapmle_rate': self.sample_rate
        }

    def morse_to_timestep_numeric(self, morse_code = "", human_dist = False, human_dist_value = 1.0, timing = None):
        numeric_sequence = []
        for char in morse_code:
            if char == '.':
                # Dot vector
                if human_dist:
                    dotduration = timing['dotduration'] * human_dist_value
                    vect = [1] * max(1, int(round(dotduration * self.sample_rate)))
                    vect += [0] * max(1,int(round(dotduration * self.sample_rate)))
                    #if len(vect) <= 6:
                    #    print(f"dotduration: {int(dotduration * self.sample_rate), dotduration * self.sample_rate, round(dotduration * self.sample_rate)} : {vect}")
                else:
                    vect = [1] * timing['dot'] + [0] * timing['intra_char_space']
                numeric_sequence.extend(vect)
            elif char == '-':
                # Dash vector
                if human_dist:
                    dashduration = timing['dashduration'] * human_dist_value
                    spaceduration = timing['space_between_elements_duration'] * human_dist_value
                    vect = [1] * max(3,int(round(dashduration * self.sample_rate)))
                    vect += [0] * max(3,int(round(spaceduration * self.sample_rate)))
                    #if len(vect) <= 12:
                    #    print(f"dashduration: {int(dashduration * self.sample_rate), dashduration * self.sample_rate, round(dashduration * self.sample_rate)} : {vect}")
                else:
                    vect = [1] * timing['dash'] + [0] * timing['intra_char_space']
                numeric_sequence.extend(vect)
            elif char == '§':  # Space between characters
                # Space between characters vector
                if human_dist:
                    charspaceduration = (timing['space_between_characters_duration'] - timing['space_between_elements_duration']) * human_dist_value
                    vect = [0] * max(15, int(round(charspaceduration * self.sample_rate)))
                    #if len(vect) <= 12:
                    #    print(f"dashduration: {int(charspaceduration * self.sample_rate), charspaceduration * self.sample_rate, round(charspaceduration * self.sample_rate)} : {vect}")
                else:
                    vect = [0] * (timing['inter_char_space'] - timing['intra_char_space'])
                numeric_sequence.extend(vect)
         
        #print(f"len(numeric_sequence): {len(numeric_sequence)}")
        print(f"numeric_sequence: {numeric_sequence}")

        return numeric_sequence

    def create_training_dataset(self, filename='struggle8_training_data.pkl'):
        # Prepare dataset with multiple characters
        X = []
        y = []

        for takt in range(150, 151, 5):
            for letter, code in self.morse_code_dict.items():
                for _ in range(5):  # Repeat each letter with perfect timing 10 times
                    X.append(self.morse_to_timestep_numeric(code + '§', human_dist=False, timing=self.generate_timing(swe_takt=takt)))
                    y.append(self.unique_identifiers[letter])
                    # Add with some human distortion
                for _ in range(10):  # Make sure to have some human distortion in the dataset. Repeat each letter 5 times with human distortions +- 20%
                    human_dist_value = random.uniform(0.8, 1.2)
                    for _ in range(3):  # Repeat each letter 10 times with human distortions +- 20%
                        X.append(self.morse_to_timestep_numeric(code + '§', human_dist=True, human_dist_value=human_dist_value, timing=self.generate_timing(swe_takt=takt)))
                        y.append(self.unique_identifiers[letter])

        # Print longest sequence of X
        x = max(X, key=len)
        xi = X.index(x)

        print(f"Longest sequence of X[{xi}]: {len(x)}")
        print(f"X[{xi}]: {x}")
        print(f"y[{xi}]: {y[xi]}")        

        # Shuffle the dataset
        combined = list(zip(X, y))
        random.shuffle(combined)
        X[:], y[:] = zip(*combined)

        # Pad sequences for consistent input size
        X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen = 550, dtype='int8',)

        # Convert output to categorical
        y = to_categorical(y, num_classes=self.num_categories)  # Ensure correct number of categories

        # Save history with pickle (only the history attribute)
        with open(filename, 'wb') as f:  # Use .pkl as the file extension for clarity
            pickle.dump([X, y], f)

        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")

# Main code optimized
if __name__ == '__main__': 
    ctraindata = MorseCodeTrainingDataGeneratorClass(sample_rate=40) # Sample rate
    ctraindata.create_training_dataset("struggle8_training_data.pkl")
    ctraindata.create_training_dataset("struggle8_test_data.pkl")

    print("Done!")
