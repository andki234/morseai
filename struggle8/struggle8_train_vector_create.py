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
            'É': '..-..', 'Ñ': '--.--', 'Ü': '..--',

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
  
    def generate_timing(self, nof_dot_samples):
        # Calculate samples per dot
        samples_per_dot = int(nof_dot_samples)
        # Calculate samples per dash
        samples_per_dash = int(nof_dot_samples * 3)
        # Calculate samples per space between elements
        samples_per_space_between_elements = int(nof_dot_samples)
        # Calculate samples per space between characters
        samples_per_space_between_characters = int(nof_dot_samples * 3)
        # Calculate samples per space between words
        samples_per_space_between_words = int(nof_dot_samples * 7)

        # Return dictionary with timing
        return {
            'dot': samples_per_dot,
            'dash': samples_per_dash,
            'intra_char_space': samples_per_space_between_elements,
            'inter_char_space': samples_per_space_between_characters,
            'inter_word_space': samples_per_space_between_words,
            'sapmle_rate': self.sample_rate
        }

    def morse_to_timestep_numeric(self, morse_code = "", human_dist = False, human_dist_value_dot = 1.0, human_dist_value_dash = 1.0, human_dist_value_element_space = 1.0, timing = None):
        numeric_sequence = []
        for char in morse_code:
            if char == '.':
                # Dot vector
                if human_dist:
                    dotduration = timing['dot'] * round(human_dist_value_dot * random.uniform(0.9, 1.1))
                    spaceduration = timing['intra_char_space'] * round(human_dist_value_element_space * random.uniform(0.9, 1.1))
                    vect = [1] * max(1, dotduration)
                    vect += [0] * max(1,spaceduration)
                else:
                    vect = [1] * timing['dot'] + [0] * timing['intra_char_space']
                numeric_sequence.extend(vect)
            elif char == '-':
                # Dash vector
                if human_dist:
                    dashduration = timing['dot'] * round(human_dist_value_dot * random.uniform(0.9, 1.1))
                    spaceduration = timing['intra_char_space'] * round(human_dist_value_element_space * random.uniform(0.9, 1.1))
                    vect = [1] * max(1, dashduration)
                    vect += [0] * max(1,spaceduration)
                else:
                    vect = [1] * timing['dash'] + [0] * timing['intra_char_space']
                numeric_sequence.extend(vect)
            elif char == '§':  # Space between characters
                # Space between characters vector
                if human_dist:
                    charspaceduration = (timing['inter_char_space'] - timing['intra_char_space']) * round(human_dist_value_dot * random.uniform(0.9, 1.1))
                    vect = [0] * max(7, charspaceduration)
                else:
                    vect = [0] * (timing['inter_char_space'] - timing['intra_char_space'])
                numeric_sequence.extend(vect)
         

        return numeric_sequence

    def create_training_dataset(self, filename='struggle8_training_data.pkl'):
        # Prepare dataset with multiple characters
        X = []
        y = []

        for nof_dot_samples in range(3, 3*10, 3):
            tim = self.generate_timing(nof_dot_samples=nof_dot_samples)
            print(f"takt: {nof_dot_samples}")
            print(f"tim: {tim}")
            for letter, code in self.morse_code_dict.items():
                for _ in range(5):  # Repeat each letter with perfect timing 10 times
                    X.append(self.morse_to_timestep_numeric(code + '§', human_dist=False, timing=self.generate_timing(nof_dot_samples=nof_dot_samples)))
                    y.append(self.unique_identifiers[letter])
                    # Add with some human distortion
                for _ in range(10):  # Make sure to have some human distortion in the dataset. Repeat each letter 5 times with human distortions +- 20%
                    human_dist_value_dot = random.uniform(1.0, 1.0)
                    human_dist_value_dash = random.uniform(1.0, 1.0)
                    human_dist_value_element_space =  random.uniform(-(tim["intra_char_space"]/4), tim["intra_char_space"]/4)
                    for _ in range(3):  # Repeat each letter 10 times with human distortions +- 20%
                        X.append(self.morse_to_timestep_numeric(code + '§', 
                                                                human_dist=True, 
                                                                human_dist_value_dot=human_dist_value_dot, 
                                                                human_dist_value_dash=human_dist_value_dash, 
                                                                human_dist_value_element_space=human_dist_value_element_space,
                                                                timing=self.generate_timing(nof_dot_samples=nof_dot_samples)))
                        y.append(self.unique_identifiers[letter])

                for _ in range(10):  # Make sure to have some human distortion in the dataset. Repeat each letter 5 times with human distortions +- 20%
                    human_dist_value_dot = random.uniform(1.0, 1.0)
                    human_dist_value_dash = random.uniform(-(tim["dash"]/4), tim["dash"]/4)
                    human_dist_value_element_space = random.uniform(1.0, 1.0)
                    for _ in range(3):  # Repeat each letter 10 times with human distortions +- 20%
                        X.append(self.morse_to_timestep_numeric(code + '§', 
                                                                human_dist=True, 
                                                                human_dist_value_dot=human_dist_value_dot, 
                                                                human_dist_value_dash=human_dist_value_dash, 
                                                                human_dist_value_element_space=human_dist_value_element_space,
                                                                timing=self.generate_timing(nof_dot_samples=nof_dot_samples)))
                        y.append(self.unique_identifiers[letter])

                for _ in range(10):  # Make sure to have some human distortion in the dataset. Repeat each letter 5 times with human distortions +- 20%
                    human_dist_value_dot = random.uniform(-(tim["dot"]/4), tim["dot"]/4)
                    human_dist_value_dash = random.uniform(1.0, 1.0)
                    human_dist_value_element_space = random.uniform(1.0, 1.0)
                    for _ in range(3):  # Repeat each letter 10 times with human distortions +- 20%
                        X.append(self.morse_to_timestep_numeric(code + '§', 
                                                                human_dist=True, 
                                                                human_dist_value_dot=human_dist_value_dot, 
                                                                human_dist_value_dash=human_dist_value_dash, 
                                                                human_dist_value_element_space=human_dist_value_element_space,
                                                                timing=self.generate_timing(nof_dot_samples=nof_dot_samples)))
                        y.append(self.unique_identifiers[letter])


                for _ in range(10):  # Make sure to have some human distortion in the dataset. Repeat each letter 5 times with human distortions +- 20%
                    human_dist_value_dot = random.uniform(-1, 1)
                    human_dist_value_dash = random.uniform(-1, 1)
                    human_dist_value_element_space = random.uniform(-1, 1)
                    for _ in range(3):  # Repeat each letter 10 times with human distortions +- 20%
                        X.append(self.morse_to_timestep_numeric(code + '§', 
                                                                human_dist=True, 
                                                                human_dist_value_dot=human_dist_value_dot, 
                                                                human_dist_value_dash=human_dist_value_dash, 
                                                                human_dist_value_element_space=human_dist_value_element_space,
                                                                timing=self.generate_timing(nof_dot_samples=nof_dot_samples)))
                        y.append(self.unique_identifiers[letter])




        # Print longest sequence of X
        x = max(X, key=len)
        xi = X.index(x)

        print(f"Longest sequence of X[{xi}]: {len(x)}")
        print(f"X[{xi}]: {x}")
        print(f"y[{xi}]: {y[xi]}")        

        # Shuffle the dataset
        #combined = list(zip(X, y))
        #random.shuffle(combined)
        #X[:], y[:] = zip(*combined)

        # Pad sequences for consistent input size
        X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen = 2000, dtype='int8',)

        # Convert output to categorical
        y_cat = to_categorical(y, num_classes=self.num_categories)  # Ensure correct number of categories

        # Save history with pickle (only the history attribute) for y_cat
        with open(filename, 'wb') as f:  # Use .pkl as the file extension for clarity
            pickle.dump([X, y_cat], f)

        # Save history with pickle (only the history attribute) for y
        with open('raw_y_' + filename, 'wb') as f:  # Use .pkl as the file extension for clarity
            pickle.dump(y, f)

        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y_cat.shape}")

# Main code optimized
if __name__ == '__main__': 
    ctraindata = MorseCodeTrainingDataGeneratorClass(sample_rate=48) # Sample rate
    ctraindata.create_training_dataset("struggle8_training_data.pkl")
    ctraindata.create_training_dataset("struggle8_test_data.pkl")

    print("Done!")
