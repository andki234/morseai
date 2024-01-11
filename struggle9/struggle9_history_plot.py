import matplotlib.pyplot as plt
import pickle

class HistoryPlot(): 
    def __init__(self):
        # Load the history dictionary with pickle
        with open('struggle9_model_history.pkl', 'rb') as f:
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

        # Plotting training and validation loss
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, accuracy_values, 'g', label='Training accuracy')
        plt.plot(epochs, val_accuracy_values, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        # Check if 'lr' is in history_dict
        if 'lr' in history_dict:
            learning_rate = history_dict['lr']
            plt.plot(epochs, learning_rate, 'b-')
            plt.title('Learning Rate over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    plotter = HistoryPlot()
