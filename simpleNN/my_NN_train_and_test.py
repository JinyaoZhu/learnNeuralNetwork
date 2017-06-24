class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,num_of_hidden_layers,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.num_of_hidden_layers = num_of_hidden_layers
        self.lr = learningrate
        
        self.wh = []
        
        for n in range(0,self.num_of_hidden_layers+1):
            if n==0:
                self.wh.append((numpy.random.rand(self.hnodes,self.inodes)-0.5))
            elif n == self.num_of_hidden_layers:
                self.wh.append((numpy.random.rand(self.onodes,self.hnodes)-0.5))
            else:
                self.wh.append((numpy.random.rand(self.hnodes,self.hnodes)-0.5))
            
        
        self.activation_function = lambda x:scipy.special.expit(x)
    

    def train(self,inputs_list,targets_list):
        
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        hidden_inputs = inputs
        
        layers_outputs = []
        layers_outputs.append(inputs)
        
        for n in range(0,self.num_of_hidden_layers+1):
            hidden_inputs = numpy.dot(self.wh[n],hidden_inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs = hidden_outputs
            layers_outputs.append(hidden_outputs)
        
        hidden_errors = targets - layers_outputs[self.num_of_hidden_layers+1]
        
        for n in range(self.num_of_hidden_layers,-1,-1):
            self.wh[n] += self.lr*numpy.dot(hidden_errors*layers_outputs[n+1]*(1.0-layers_outputs[n+1]),numpy.transpose(layers_outputs[n]))
            hidden_errors = numpy.dot(self.wh[n].T,hidden_errors)
            
        #self.output_errors = output_errors
        #print(output_errors)
        #return output_errors
    
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = inputs
        
        for n in range(0,self.num_of_hidden_layers+1):
            hidden_inputs = numpy.dot(self.wh[n],hidden_inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs = hidden_outputs
            
        return hidden_outputs

import numpy
import scipy.special
import matplotlib.pyplot
import dill
#%matplotlib inline

input_nodes = 784
output_nodes = 10

hidden_nodes = 200
num_hidden_layers = 1
learning_rate = 0.005
epoch = 5

nn = neuralNetwork(input_nodes,hidden_nodes,output_nodes,num_hidden_layers,learning_rate)

print("Reading data...\n")
train_data_file = open("/home/jinyao/Downloads/mnist_train.csv",'r')
train_data_list = train_data_file.readlines()
train_data_file.close()
print("Trainning...\n")
for n in range(0,epoch):
    print("Epoch:",epoch)
    for record in train_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])]=0.99
        nn.train(inputs,targets)

print("Trainning finish!")

print("Saving model...")
dill.dump(nn, open('model.pkl', 'wb'))

print("Loading test data...")
test_data_file = open("/home/jinyao/Downloads/mnist_test.csv",'r')
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
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print("=============================================")
print("Model parameters:")
print("---------------------------------------------")
print("hidden nodes:",nn.hnodes)
print("hidden layers:",nn.num_of_hidden_layers)
print("learning rate:",nn.lr)
print("=============================================")
print("performance = ",scorecard_array.sum()/scorecard_array.size)
print("=============================================")