import numpy
import scipy.special
import matplotlib.pyplot
import dill
import os

print("Loading model...")
nn = dill.load(open('model.pkl', 'rb'))

print("Loading test data...")
test_data_file = open("/home/jinyao/Downloads/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

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
print("=============================================")
print("Model parameters:")
print("---------------------------------------------")
print("hidden nodes:", nn.hnodes)
print("hidden layers:", nn.num_of_hidden_layers)
print("learning rate:", nn.lr)
print("=============================================")
print("performance = ", scorecard_array.sum()/scorecard_array.size)
print("=============================================")
