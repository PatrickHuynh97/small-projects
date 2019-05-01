import matplotlib.pyplot as plt


def plot_history(history, acc=True, loss=True, val=False):
    # plot training and validation accuracy over epochs
    if acc:
        plt.plot(history.history['acc'])
        if val:
            plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    # plot training and validation loss over epochs
    if loss:
        plt.plot(history.history['loss'])
        if val:
            plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
