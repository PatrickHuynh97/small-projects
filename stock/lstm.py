from random import choice, choices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler


def get_data(not_comp=None, comp=None):
    """
    Gets historical stock data for one company in S&P 500 index for last 5 years in
    train/test split
    :param not_comp: if given, dthis company is ignored
    :param comp: company to get stock prices for. If not given, random one is chosen
    :return: train/test data set and company chosen
    """
    stock_data = pd.read_csv('all_stocks_5yr.csv')

    if comp:
        stock_data = stock_data[stock_data['Name'] == comp]
    else:
        companies_ = stock_data[stock_data['Name'] != not_comp].Name.unique()
        comp = choice(companies_)
        stock_data = stock_data[stock_data['Name'] == comp]

    return stock_data, comp


def get_companies():
    # get all companies as list
    stock_data = pd.read_csv('all_stocks_5yr.csv')
    return list(stock_data.Name.unique())


def to_2d_array(array):
    return np.reshape(array, (-1, 1))


def scale_data(data, scaler, fit):
    """
    Scales 1D array of labels to 2D MinMax scaled labels
    :param data: 1D array of labels
    :param scaler: scaler to use
    :param fit: if true, scaler is also fit to data
    :return: 2D array of scaled labels
    """
    labels_2d = to_2d_array(data)  # make labels 2D array for MinMaxScaler
    if fit:
        return scaler.fit_transform(labels_2d)
    else:
        return scaler.transform(labels_2d)


def reshape_data(data, number_of_samples, timestep, features):
    return np.reshape(data, (number_of_samples, timestep, features))


def format_data(data, ts=None):
    """
    Takes training set of data and splits it into x_train and y_train. A single training sample for date X is
    'close' values from dates [X-tp:X], and a label of the 'close' value for date X+1.
    :param data: historic data to train on
    :param ts: number of data samples to use for a single label
    :return: training data and corresponding labels
    """
    if not ts:
        ts = 30  # if no time period is defined, set to 30 days
    x = []
    y = []
    length = len(data)
    for i in range(ts, length):
        x.append(data[i - ts:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


def test_model(trained_model, trained_comp, ts, scaler, test_comp=None):
    """
    Tests model performance on randomly chosen company from stock data
    :param trained_model: train model
    :param trained_comp: company model was trained on
    :param ts: time step used during training
    :param scaler: scaler used during training
    :param test_comp: company to test model on
    :return:
    """
    # get stock data for a random company, ignoring company with name comp (was used for training)
    data, test_comp = get_data(not_comp=trained_comp, comp=test_comp)

    # y_test is actual stock prices from the nth day where n is the number of time steps before n, used to predict n
    y_test = to_2d_array(data['Close'].values[ts:])

    # x_test is the same as before, but we ignore the labels this time as we are predicting them using the model
    scaled_data = scale_data(data['Close'].values, scaler, fit=False)
    x_test, _ = format_data(scaled_data, ts=ts)

    x_test = reshape_data(data=x_test, number_of_samples=x_test.shape[0], timestep=ts, features=1)

    # use model to predict closing price for each sample in x_test
    y_predictions = trained_model.predict(x_test)

    # undo the MinMax scaling done before prediction to get actual predictions
    predicted_close = scaler.inverse_transform(y_predictions)

    # plot
    plt.plot(y_test, color='green', label='Actual Price')
    plt.plot(predicted_close, color='red', label='Predicted Price')
    plt.title('LSTM trained on {}, predicting stock for {}'.format(trained_comp, test_comp))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # get historic data for a random company
    data_df, training_company = get_data()

    print("Predicting stock prices for {}".format(training_company))

    sc = MinMaxScaler()

    # get training set by scaling 'Close' values for data set
    training_set = scale_data(data=data_df['Close'].values, scaler=sc, fit=True)

    time_step = 14

    # split training set into data and labels, where data is tp days of 'close' and label is the next day's close
    x_train, y_train = format_data(training_set, ts=time_step)

    # reshape data to three dimensions as required for input to LSTM (no_samples, time_step, features). In this
    # instance, time_step is defined above as 14 days, and we have 1 feature (close). The number of samples depends on
    # which company was chosen
    no_samples = x_train.shape[0]
    x_train = reshape_data(x_train, number_of_samples=no_samples, timestep=time_step, features=1)

    # define LSTM
    # Model taken from https://www.kaggle.com/pankul/lstm-stock-price-movement-prediction/notebook
    model = Sequential()

    model.add(LSTM(units=92, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=92, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=92, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=92, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')

    # train model
    history = model.fit(x_train, y_train, epochs=30, batch_size=32)

    companies = get_companies()
    companies.remove(training_company)

    # test model on 3 different random companies
    for company in choices(companies, k=10):
        test_model(trained_model=model,
                   trained_comp=training_company,
                   test_comp=company,
                   ts=time_step,
                   scaler=sc)
