import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_dataframe(df, test_size):
    """
    Split dataframe into shuffled train/test set, requires "labels" column
    :param df: dataframe
    :param test_size: size of test set
    :return: train/test DataFrames with one-hot encoded labels
    """
    # split into train/test data
    train, test = train_test_split(df, shuffle=True, test_size=test_size)

    x_train = train.drop(columns=['labels'])
    y_train = pd.get_dummies(train['labels'])  # one-hot encoded labels

    x_test = test.drop(columns=['labels'])
    y_test = pd.get_dummies(test['labels'])  # one-hot encoded labels

    return x_train, y_train, x_test, y_test
