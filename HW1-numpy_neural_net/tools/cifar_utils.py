import pickle
import numpy as np
import os

def load_CIFAR10_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        features = batch[b'data'].astype('float')
        labels = batch[b'labels']
        labels = np.array(labels)
        return features, labels

def load_CIFAR10(cifar_path, batches_num=6):
    X_train = []
    Y_train = []
    for batch in range(1, batches_num):
        file = os.path.join(cifar_path, 'data_batch_{}'.format(batch))
        features, labels = load_CIFAR10_batch(file)
        X_train.append(features)
        Y_train.append(labels)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_test, Y_test = load_CIFAR10_batch(os.path.join(cifar_path, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

