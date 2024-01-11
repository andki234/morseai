import numpy as np
import random

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

morse_code_dict_A = {
        'A': '.-',  'B': '-...', 'C': '-.-.', 'D': '-..', 
        'E': '.',  'F': '..-.', 'G': '--.', 'H': '....',
        'I': '..', 'J': '.---', 'K': '-.-',  'L': '.-..',
        'N': '-.', 'P': '.--.',
        'Q': '--.-', 'R': '.-.',  'S': '...', 
        'U': '..-',  'V': '...-', 'W': '.--',  'X': '-..-',
        'Y': '-.--', 'Z': '--..',
}

# Create a unique numerical identifier for each Morse code symbol
unique_identifiers = {symbol: i for i, symbol in enumerate(morse_code_dict.keys())}
num_categories = len(unique_identifiers)

def generate_timing(nof_dot_samples):
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
            'inter_word_space': samples_per_space_between_words
        }

def morse_to_timestep_numeric(morse_code = "", human_dist = False, human_dist_value_dot = 0.0, human_dist_value_dash = 0.0, human_dist_value_element_space = 0.0, timing = None):
        numeric_sequence = []
        for char in morse_code:
            if char == '.':
                # Dot vector
                if human_dist:
                    dotduration = timing['dot'] + round(human_dist_value_dot * random.uniform(-1.0, 1.0))
                    spaceduration = timing['intra_char_space'] + round(human_dist_value_element_space * random.uniform(-1.0, 1.0))
                    vect = [1] * max(3, dotduration)
                    vect += [0] * max(3,spaceduration)
                else:
                    vect = [1] * timing['dot'] + [0] * timing['intra_char_space']
                numeric_sequence.extend(vect)
            elif char == '-':
                # Dash vector
                if human_dist:
                    dashduration = timing['dash'] + round(human_dist_value_dash * random.uniform(-1.0, 1.0))
                    spaceduration = timing['intra_char_space'] + round(human_dist_value_element_space * random.uniform(-1.0, 1.1))
                    vect = [1] * max(3*3, dashduration)
                    vect += [0] * max(3*3,spaceduration)
                else:
                    vect = [1] * timing['dash'] + [0] * timing['intra_char_space']
                numeric_sequence.extend(vect)
            elif char == '§':  # Space between characters
                # Space between characters vector
                if human_dist:
                    charspaceduration = (timing['inter_char_space'] - timing['intra_char_space']) + round(human_dist_value_dot * random.uniform(-1.0, 1.1))
                    vect = [0] * max(3*7, charspaceduration)
                else:
                    vect = [0] * (timing['inter_char_space'] - timing['intra_char_space'])
                numeric_sequence.extend(vect)
         

        return numeric_sequence