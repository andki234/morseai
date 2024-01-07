import matplotlib.pyplot as plt
import pickle

class HistoryPlot(): 
    def __init__(self):
        # Load the history dictionary with pickle
        with open('struggle7_model_history.pkl', 'rb') as f:
            history = pickle.load(f)
        
        # Plot the history
        self.plot_history(history)
    
    def plot_history(self, history):
        # history is already a dictionary
        history_dict = history

        # Extracting loss and accuracy for training and validation sets
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        accuracy_values = history_dict['accuracy']
        val_accuracy_values = history_dict['val_accuracy']
        epochs = range(1, len(loss_values) + 1)

        plt.figure(figsize=(18, 6))  # Adjust the figure size as needed

        # Plotting training and validation loss
        plt.subplot(1, 3, 1)  # Changed to 1 row, 3 columns, position 1
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and validation accuracy
        plt.subplot(1, 3, 2)  # Changed to 1 row, 3 columns, position 2
        plt.plot(epochs, accuracy_values, 'g', label='Training accuracy')
        plt.plot(epochs, val_accuracy_values, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plotting learning rate
        if 'lr' in history_dict:
            learning_rate = history_dict['lr']
            plt.subplot(1, 3, 3)  # Added as the third subplot
            plt.plot(epochs, learning_rate, 'b-')
            plt.title('Learning Rate over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')

        plt.tight_layout()  # Adjust layout to fit the new subplot
        plt.show()
      

if __name__ == '__main__':
    plotter = HistoryPlot()
