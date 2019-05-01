import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from sklearn.decomposition import PCA
import pandas as pd

from breast_cancer.helpers import get_std_cancer_data
from utils.plotting import plot_history
from utils.preprocessing import train_test_split_dataframe


def pca_nn_breast_cancer(n_comp, model=None):
    """
    Functions performs PCA on breast cancer data set and uses a simple Neural Network as a classifier
    :param n_comp: number of components to use for PCA
    :param model: optional model, if not provided a hardcoded model will be used
    """
    # Load standardized data set
    data_std, labels = get_std_cancer_data()

    # project standardized data onto its n_comp principal components
    pca = PCA(n_components=n_comp).fit_transform(data_std)

    columns = ['PC{}'.format(i+1) for i in range(n_comp)]
    columns.append('labels')

    # put 3 principal components into dataframe with labels
    data_df = pd.DataFrame(data=np.append(pca, np.reshape(labels, [-1, 1]), axis=1),
                           columns=columns)
    data_df['labels'] = data_df['labels'].astype(int)

    x_train, y_train, x_test, y_test = train_test_split_dataframe(data_df, 0.15)

    # define and compile network
    if not model:
        model = Sequential()
        model.add(Dense(n_comp, activation='relu', input_dim=x_train.shape[1]))  # first layer has n_comp nodes
        model.add(Dense(18, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #  early stopping monitor so the model stops training when no more changes are being made
    early_stopping_monitor = EarlyStopping(patience=3)

    # train model
    history = model.fit(x_train, y_train,
                        validation_data=[x_test, y_test],
                        epochs=30,
                        callbacks=[early_stopping_monitor])

    plot_history(history, val=True)

    return history.history['val_loss'][-1], history.history['val_acc'][-1]


if __name__ == "__main__":

    # run pca_nn with default model
    n_components = 3
    def_loss, def_acc = pca_nn_breast_cancer(n_components)  # run PCA (n_components = 4) then feed through NN

    # define new model, run pca_nn again with new model
    n_components = 5
    model = Sequential()
    model.add(Dense(5, activation='relu', input_dim=n_components))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cus_loss, cus_acc = pca_nn_breast_cancer(n_components, model=model)

    print()
    print("Model 1 loss: {}, accuracy: {}".format(def_loss, def_acc))
    print("Model 2 loss: {}, accuracy: {}".format(cus_loss, cus_acc))
