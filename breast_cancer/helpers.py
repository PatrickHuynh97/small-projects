from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def get_std_cancer_data():
    # Load data set, standardize (mean = 0, variance = 1), then return
    cancer = load_breast_cancer()
    target = cancer.target
    return StandardScaler().fit_transform(cancer.data), target


def get_cancer_data():
    # Load data set then return
    cancer = load_breast_cancer()
    target = cancer.target
    return cancer.data, target
