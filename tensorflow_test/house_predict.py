from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

import os
#ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

tf.logging.set_verbosity(tf.logging.ERROR)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]

FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]

LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)

feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir="/tmp/boston_model")

def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values,shape=[data_set[k].size,1])
                    for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values,shape=[data_set[LABEL].size,1])
    return feature_cols, labels


regressor.fit(input_fn=lambda: input_fn(training_set), steps=10000)

ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)

loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

y = regressor.predict_scores(input_fn=lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, 6))
print ("Predictions: {}".format(str(predictions)))