import sys
import argparse
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def classify_naive_bayes(train_x,train_y,test_x,test_y):
    return 0


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description='Classify the data.')
    parser.add_argument('--methods', help='delimited list input', type=str, default="naive_bayes,svn,decision_tree,random_forest,adaboost")

    args = parser.parse_args()
    methods = [item for item in args.methods.split(',')]
    print(methods)
    data = pd.read_csv("../data/train.csv")
    train_x,test_x,train_y,test_y = train_test_split(data.drop("activity",axis=1),data[['activity']],train_size=0.75,random_state=123)