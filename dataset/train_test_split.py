# Split the dataset in training set and testing set in order to be consistent on the comparisons
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Iris.csv', index_col=0)

SEED = 1

# 80/20 holdout split
train, test = train_test_split(df, stratify=df['Species'], test_size=30, random_state=SEED)

train.to_csv('Iris_training.csv')
test.to_csv('Iris_testing.csv')