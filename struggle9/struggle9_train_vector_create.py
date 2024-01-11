import struggle9_base as base
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class MorseCodeTrainingDataGeneratorClass:
    def __init__(self):
        print("MorseCodeTrainingDataGeneratorClass.__init__()")
        
        
    def create_training_dataset(self, human_dist = 0.1, start = 3, step = 3, end = 30, maxlen = 100, is_test_vector = False, filename='struggle9_training_data.pkl'):
        # Prepare dataset with multiple characters
        X = []
        y = []
        
        # Create training data
        for nof_dot_samples in range(start, end+1, step):
            tim = base.generate_timing(nof_dot_samples=nof_dot_samples)
            print(f"tim: {tim}")
            # Convert scalars to 1-dimensional arrays
            dot_array = np.array([tim['dot']])
            dash_array = np.array([tim['dash']])
            intra_char_space_array = np.array([tim['intra_char_space']])
            inter_char_space_array = np.array([tim['inter_char_space']])
            inter_word_space_array = np.array([tim['inter_word_space']])
            for letter, code in base.morse_code_dict.items():
               
                for _ in range(10):  # Repeat each letter with perfect timing 10 times without wpm
                    combined_vector =np.array(base.morse_to_timestep_numeric(code, human_dist=False, timing=tim))
                    X.append(combined_vector.flatten())
                    y.append(base.unique_identifiers[letter])
                    
                for _ in range(50):  # Repeat each letter with perfect timing 10 times
                    #if is_test_vector:
                    combined_vector =np.array(base.morse_to_timestep_numeric(code, human_dist=False, timing=tim))
                    #else:
                    #    combined_vector = np.concatenate((base.morse_to_timestep_numeric(code, human_dist=False, timing=tim), 
                    #                                      dot_array, intra_char_space_array, dash_array, inter_char_space_array, inter_word_space_array))
                    X.append(combined_vector.flatten())
                    y.append(base.unique_identifiers[letter])
                    
        
        
        # Find the index of the longest vector in X
        index_of_longest = np.argmax([len(vector) for vector in X])
        
        print(f"Longest sequence of X[{index_of_longest}]: {len(X[index_of_longest])}")
        print(f"X[{index_of_longest}]: {X[index_of_longest]}")
        print(f"y[{index_of_longest}]: {y[index_of_longest]}")        

        # Shuffle the dataset
        #combined = list(zip(X, y))
        #random.shuffle(combined)
        #X[:], y[:] = zip(*combined)
        
        # Pad sequences for consistent input size
        X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen = maxlen, dtype='float32')
    
        # Convert output to categorical
        y_cat = to_categorical(y, num_classes=base.num_categories)  # Ensure correct number of categories

        # Save history with pickle (only the history attribute) for y_cat
        with open(filename, 'wb') as f:  # Use .pkl as the file extension for clarity
            pickle.dump([X, y_cat], f)

        # Save history with pickle (only the history attribute) for y
        with open('raw_y_' + filename, 'wb') as f:  # Use .pkl as the file extension for clarity
            pickle.dump(y, f)

        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y_cat.shape}")
          
    
# Main code 
if __name__ == '__main__': 
    ctraindata = MorseCodeTrainingDataGeneratorClass() 
    ctraindata.create_training_dataset(human_dist = 0.1, start = 3, step = 3, end = 30, maxlen = 620, is_test_vector=False, filename = "struggle9_training_data.pkl")
    ctraindata.create_training_dataset(human_dist = 0.1, start = 4, step = 3, end = 31, maxlen = 620, is_test_vector=True, filename = "struggle9_test_data.pkl")

    print("Done!")
