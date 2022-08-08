import random
import numpy as np
from sklearn.model_selection import KFold

def wavelet_transform(mat):
    print("Not implemented.")
    return 1;

def train()

def cross_val_train(X, Y, model, seed=0):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)


def load_data(npz_contents, wavelet, seed=0):
    X, Y, stack_indices, stack_names = (npz_contents['X'], npz_contents['Y'], npz_contents['stack_indices'], npz_contents['stack_names'])

    if wavelet:
        X = wavelet_transform(X)
        Y = wavelet_transform(Y)

    sample_range = len(stack_indices)

    random.seed(seed)
    train_val_indices = sorted(random.sample(range(0, sample_range), sample_range - 6))
    test_indices = [index for index in range(sample_range) if index not in train_val_indices]

    X_train, Y_train, X_test, Y_train = ..., ..., ..., ...

