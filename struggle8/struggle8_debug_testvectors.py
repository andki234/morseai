import tkinter as tk
from tkinter.font import Font
import pandas as pd  # or import numpy as np, depending on your data format
import pickle
import numpy as np
import struggle8_base as s8b

def show_test_vectors():
    X_train, y_train = load_test_vectors()

    # Convert y_train to a numpy array
    y_train = np.array(y_train)
    
    char_idx = s8b.unique_identifiers['B']

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
    test_vectors = [vector.rstrip('0') for vector in test_vectors]

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
    with open('struggle8_training_data.pkl', 'rb') as f:  # Use .pkl as the file extension for clarity
        X_train, y_train = pickle.load(f)
    with open('raw_y_struggle8_training_data.pkl', 'rb') as f:  # Use .pkl as the file extension for clarity
        y_train = pickle.load(f)
    return X_train, y_train

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
custom_font = Font(family="Consolas", size=16)

# Create a text area and apply the custom font
text_area = tk.Text(window, font=custom_font, height=50, width=300)
text_area.pack()

# Start the GUI event loop
window.mainloop()
