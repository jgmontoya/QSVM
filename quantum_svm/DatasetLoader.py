import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler


def normalize_and_scale(training, testing, features, gaussian=True, minmax=False):
    # Normalizes and/or scales both datasets based on the training dataset.
    
    sample_train = training.copy()
    sample_test = testing.copy()
    
    if gaussian:
        # Gaussian around 0 with unit variance normalization
        for feature in features:
            mean = sample_train[feature].mean()
            std = sample_train[feature].std()
            sample_train[feature] = (sample_train[feature] - mean)/std
            sample_test[feature] = (sample_test[feature] - mean)/std
    
    if minmax:
        # Scale features to the range (0,1)
        for feature in features:
            min_val = sample_train[feature].min()
            max_val = sample_train[feature].max()
            dif = (max_val - min_val)
            sample_train[feature] = (sample_train[feature] - min_val)/dif
            sample_test[feature] = (sample_test[feature] - min_val)/dif

    return sample_train, sample_test

def LoadDataset(training_path, testing_path, features, label, gaussian=True, minmax=False):
    '''Loads the data, normalizes it and returns it in the following format:
    {class_0: points_0, class_1:points_1, ...}
    Where points_i corresponds to the points that belong to class_i as a numpy array
    '''
    df_train = pd.read_csv(training_path, index_col=0)
    df_test = pd.read_csv(testing_path, index_col=0)
    
    train, test = normalize_and_scale(df_train, df_test, features)
    
    train_dict, test_dict = {}, {}
    for category in train[label].unique():
        train_dict[category] = train[train['Species'] == category][features].values
        test_dict[category] = test[test['Species'] == category][features].values
    
    return train_dict, test_dict
    
# Example
# features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# LoadDataset('../dataset/Iris_training.csv', '../dataset/Iris_testing.csv', features, label='Species')

