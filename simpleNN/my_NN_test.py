import numpy
import scipy.special
import matplotlib.pyplot
import dill
import os
import sys

print (sys.version)

print("Loading model...")
with open('model.pkl', 'rb') as model_file:
    nn = dill.load(model_file)

print("Loading test data...")
with open("/home/jinyao/NN_dataset/mnist_csv/mnist_test.csv", 'r') as test_data_file:
    test_data_list = test_data_file.readlines()

print("Testing model...")
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    #print(correct_label,"correct label")
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs = nn.query(inputs)
    label = numpy.argmax(outputs)
    #print(label,"network's answer")
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
accuracy = scorecard_array.mean()
print("=============================================")
print("Model parameters:")
print("---------------------------------------------")
print("hidden nodes:", nn.hnodes)
print("hidden layers:", nn.num_of_hidden_layers)
print("learning rate:", nn.lr)
print("=============================================")
print("accuracy = {}%". format(accuracy*100.0))
print("=============================================")
