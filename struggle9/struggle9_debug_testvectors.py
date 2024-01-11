import tkinter as tk
from tkinter.font import Font
import pandas as pd  # or import numpy as np, depending on your data format
import pickle
import numpy as np
import struggle9_base as base

def find_key(value, dictionary):
    return next((key for key, val in dictionary.items() if val == value), "Key not found")

def vector_to_text_file(filename):
    # Load the training data
    X_train, y_train = load_test_vectors()

    # Save all vectors to file like "Y_train:X_train"
    with open(filename, 'w') as f:
        for i in range(len(X_train)):
            # Create string vector from the binary vector
            vector = ''.join([str(int(bit)) for bit in X_train[i]])
            # Remove trailing zeros
            vector = vector.rstrip('0')
            # Write the vector to a file
            letter = find_key(y_train[i], base.unique_identifiers)
            f.write(f"{letter}:{vector}\n")

def show_test_vectors():
    X_train, y_train = load_test_vectors()

    # Convert y_train to a numpy array
    y_train = np.array(y_train)
    
    char_idx = base.unique_identifiers['A']

    # Get all index positions where the training character is 'A' from y_train (the training labels)
    a_indices = np.where(y_train == char_idx)[0]
    # Print all found indices
    print(f"Number of indices for 'A': {len(a_indices)}")
    print(f"First index for 'A': {a_indices[0]}")
    print(f"Last index for 'A': {a_indices[-1]}")

    # Copy all indeces from X_train (the training data) where the training character is 'A'
    X_vectors = X_train[a_indices]

    # Create string vectors from the binary vectors
    test_vectors = []
    for vector in X_vectors:
        test_vectors.append(''.join([str(int(bit)) for bit in vector]))

    #remove trailing zeros
    #test_vectors = [vector.rstrip('0') for vector in test_vectors]

    # sort on string length
    test_vectors.sort(key=len)

    print(X_vectors.shape)
    print(X_vectors)

    character = entry.get()
    # Assuming you have a function that returns test vectors for a given character
    #test_vectors = get_test_vectors(character)
    text_area.delete('1.0', tk.END)

    # put all test vectors in the text area a new line for each vector
    for vector in test_vectors:
        text_area.insert(tk.END, vector + '\n')
        

    print("done")

def load_test_vectors():
    with open('struggle9_training_data.pkl', 'rb') as f:  # Use .pkl as the file extension for clarity
        X_train, y_train = pickle.load(f)
    with open('raw_y_struggle9_training_data.pkl', 'rb') as f:  # Use .pkl as the file extension for clarity
        y_train = pickle.load(f)
    return X_train, y_train

# Save all 
vector_to_text_file('struggle9_test_vectors.txt')

# Set up the main window
window = tk.Tk()
window.title("Morse Code Test Vectors")

# Create an entry widget
entry_label = tk.Label(window, text="Enter a Character:")
entry_label.pack()
entry = tk.Entry(window)
entry.pack()

# Create a button widget
button = tk.Button(window, text="Get Test Vectors", command=show_test_vectors)
button.pack()

# create a custom font
custom_font = Font(family="Consolas", size=8)

# Create a text area and apply the custom font
text_area = tk.Text(window, font=custom_font, height=50, width=250)
text_area.pack()

# Start the GUI event loop
window.mainloop()
