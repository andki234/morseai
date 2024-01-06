import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
from itertools import product
import random
import pickle

# Morse code dictionary for training
morse_code_dict = {
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

# Create a unique numerical identifier for each Morse code symbol
unique_identifiers = {symbol: i for i, symbol in enumerate(morse_code_dict.keys())}
num_categories = len(unique_identifiers)

 # Count morse units for a character
def count_morse_units(char):
    if char.lower() in morse_code_dict:
        morse_sequence = morse_code_dict[char.lower()]
        #print(f"morse_sequence: {morse_sequence} : {len(morse_sequence)}")
        #print(f"sum: {sum(1 if symbol == '.' else 3 for symbol in morse_sequence)}")
        return sum(1 if symbol == '.' else 3 for symbol in morse_sequence) + len(morse_sequence) - 1
    return 0

# Count morse units in a string
def count_morse_units_in_string(s):
    total_units = 0

    for char in s:
        # Count units for the current character
        total_units += count_morse_units(char)

        # Add 7 units for space
        if char.isspace():
            total_units += 7

    if (s[len(s) - 1].isspace()):
        total_units += 3 * (len(s) - 2)  # Add 3 units for space between characters, except for the last character if it is a space
    else:
        total_units += 3 * (len(s) - 1) # Add 3 units for space between characters

    return total_units

def generate_timing(swe_takt=48, sample_rate = 40):
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
    no_units_in_paris_inc_space_between_words = count_morse_units_in_string('PARIS ')

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
    samples_per_dot = int(dot_duration_in_seconds * sample_rate)
    # Calculate samples per dash
    samples_per_dash = int(dash_duration_in_seconds * sample_rate)
    # Calculate samples per space between elements
    samples_per_space_between_elements = int(space_between_elements_in_seconds * sample_rate)
    # Calculate samples per space between characters
    samples_per_space_between_characters = int(space_between_characters_in_seconds * sample_rate)
    # Calculate samples per space between words
    samples_per_space_between_words = int(space_between_words_in_seconds * sample_rate)

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
        'inter_word_space': samples_per_space_between_words
    }

def morse_to_timestep_numeric(morse_code, timing):
    numeric_sequence = []
    for char in morse_code:
        if char == '.':
            # Dot vector
            vect = [1] * timing['dot'] + [0] * timing['intra_char_space']
            numeric_sequence.extend(vect)
        elif char == '-':
            # Dash vector
            vect = [1] * timing['dash'] + [0] * timing['intra_char_space']
            numeric_sequence.extend(vect)
        elif char == '§':  # Space between characters
            # Space between characters vector
            vect = [0] * (timing['inter_char_space'] - timing['intra_char_space'])
            numeric_sequence.extend(vect)
        

    return numeric_sequence 


# Load the model
model = load_model('model_epoch_47_loss_0.01.keras')

print(model.summary())


# Function to decode Morse code sequences
def decode_morse_sequence(morse_sequence, takt, model, unique_identifiers):
    decoded_sequence = ''
    for morse_char in morse_sequence.split(' '):
        if morse_char:  # Avoid empty strings
            numeric_sequence = morse_to_timestep_numeric(morse_char, generate_timing(swe_takt=takt))
            #print(f"numeric_sequence: {numeric_sequence}")
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([numeric_sequence], maxlen=462, padding='post')
            prediction = model.predict(np.array(padded_sequence).reshape(1, -1, 1), verbose=0)
            predicted_index = np.argmax(prediction)
            
            # Find the corresponding Morse code symbol for the predicted index
            predicted_symbol = [symbol for symbol, index in unique_identifiers.items() if index == predicted_index][0]
            decoded_sequence += predicted_symbol

    return decoded_sequence

def calculate_decoding_error(combinations, morse_code_dict, model, unique_identifiers):
    # Generate Morse code combinations
    morse_code_combinations = [' '.join([morse_code_dict[char] for char in combo]) for combo in combinations]
    error_list = []

    total = len(combinations)
    errors = 0

    for original, morse_code in zip(combinations, morse_code_combinations):
        # Decode Morse code
        decoded_sequence = decode_morse_sequence(morse_code, model, unique_identifiers)

        # Compare with the original
        if decoded_sequence != original:
            errors += 1
            error_rate = errors / total
            print(f"Decoded Morse Code '{morse_code}' into '{decoded_sequence}' instead of '{original}'")
            error_list.append((morse_code, decoded_sequence, original))
        else:
            print(f"Error rate: {errors / total:.2%}")
            print(f"Decoded Morse Code '{morse_code}' into '{decoded_sequence}'")
 
    # Calculate error rate
    error_rate = errors / total
    return error_rate, error_list

def generate_combinations_chunk(characters, start_length, end_length):
        """Generate combinations for a chunk of lengths."""
        return [''.join(comb) for length in range(start_length, end_length + 1) for comb in product(characters, repeat=length)]

characters = [
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
                    'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','.', ',', '?', '\'', '!', '/', '(', ')',
                    'Å', 'Ä', 'Ö', 'É', 'Ñ', 'Ü', '&', ':', ';', '=', '+', '-', '_', '"', '$', '@'                                
            ]

# Generate all possible combinations of characters
combinations = generate_combinations_chunk(characters, 4, 4)
# Generate morse code for all combinations

# Example Morse code for multiple characters
morse_code_example = '... -- ..... ...- -.. ..-'

# Decode Morse code
for takt in range(20, 180, 1):
    print(f"takt: {takt}")
    #print(f"morse_code_example: {morse_code_example}")
    decoded_sequence = decode_morse_sequence(morse_code_example, takt, model, unique_identifiers)
    print(f"Decoded Morse Code '{morse_code_example}' into '{decoded_sequence}'")

exit(0)

error_rate, error_list = calculate_decoding_error(combinations, morse_code_dict, model, unique_identifiers)
print(f"Decoding Error Rate: {error_rate:.2%}")
print(f"Number of errors: {len(error_list)}")
print(f"Number of combinations: {len(combinations)}")
print(f"Number of combinations with errors: {len(error_list) / len(combinations):.2%}")
print(f"Error list: {error_list}")
