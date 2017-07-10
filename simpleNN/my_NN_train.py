import numpy
import scipy.special
import matplotlib.pyplot as plt
import dill
import time
import sys
import datetime

import warnings
warnings.filterwarnings("ignore", ".*GUI is implemented.*")

print (sys.version)

plt.style.use("ggplot")


class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,num_of_hidden_layers,learningrate=0.005):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.num_of_hidden_layers = num_of_hidden_layers
        self.lr = learningrate
        self.activation_function = lambda x:scipy.special.expit(x)
        #self.activation_function = lambda x:(numpy.tanh(x)*0.5+0.5)

        self.wh = []
        self.bh = [] 
        
        #initialize the weigh
        for n in range(0,self.num_of_hidden_layers+1):
            if n==0:
                self.wh.append((numpy.random.rand(self.hnodes, self.inodes)-0.5)*0.8)
                self.bh.append((numpy.random.rand(self.hnodes, 1)-0.5)*0.8)
            elif n == self.num_of_hidden_layers:
                self.wh.append((numpy.random.rand(self.onodes, self.hnodes)-0.5)*0.8)
                self.bh.append((numpy.random.rand(self.onodes, 1)-0.5)*0.8)
            else:
                self.wh.append((numpy.random.rand(self.hnodes, self.hnodes)-0.5)*0.8)
                self.bh.append((numpy.random.rand(self.hnodes, 1)-0.5)*0.8)

    def train(self,inputs_list,targets_list):
        
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        hidden_inputs = inputs
        
        layers_outputs = []
        layers_outputs.append(inputs)
        
        #forward
        for n in range(0,self.num_of_hidden_layers+1):
            hidden_inputs = numpy.dot(self.wh[n],hidden_inputs) + self.bh[n]
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs = hidden_outputs
            layers_outputs.append(hidden_outputs)
        
        local_grad = layers_outputs[self.num_of_hidden_layers+1] - targets

        self.error = 0.5*local_grad**2;
        #backward
        for n in range(self.num_of_hidden_layers,-1,-1):
            activation_prime = layers_outputs[n+1]*(1.0-layers_outputs[n+1])
            self.wh[n] += -self.lr*numpy.dot(local_grad*activation_prime,layers_outputs[n].T)
            self.bh[n] += -self.lr*local_grad*activation_prime
            local_grad = numpy.dot(self.wh[n].T,local_grad)

        #print(output_errors)
        #return output_errors
    
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = inputs
        
        for n in range(0,self.num_of_hidden_layers+1):
            hidden_inputs = numpy.dot(self.wh[n], hidden_inputs) + self.bh[n]
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_inputs = hidden_outputs
            
        return hidden_outputs

input_nodes = 784
output_nodes = 10

hidden_nodes = 5
num_hidden_layers = 1
learning_rate = 0.01
epoch = 5

nn = neuralNetwork(input_nodes,hidden_nodes,output_nodes,num_hidden_layers,learning_rate)

print("Reading data...\n")
with open("/home/jinyao/NN_dataset/mnist_csv/mnist_train.csv",'r') as train_data_file:
    train_data_list = train_data_file.readlines()   

plt.title('Loss')
plt.xlabel('number of data / k')
plt.ylabel("Loss")
plot_loss = []

X = []
y = []
print("Preprocessing...")
for record in train_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])]=0.99
    X.append(inputs)
    y.append(targets)

plot_cnt = 0
t0 = time.clock()

print("Trainning...\n")
for n in range(0,epoch):
    print("Epoch:",n+1)
    for i in range(len(X)):
        nn.train(X[i], y[i])
        if plot_cnt%999 == 0:
            plot_loss.append(nn.error.mean())
            plt.xlim(0,len(plot_loss))
            plt.ylim(min(plot_loss)*0.9,max(plot_loss)*1.1)
            plt.plot(plot_loss,color='r')
            plt.pause(1e-5)
        plot_cnt = plot_cnt + 1

    print("Final Loss:",plot_loss[-1])

print("time cost:", datetime.timedelta(seconds=(time.clock()-t0)))
print("Trainning finish!")
print("Saving model...")

with  open('model.pkl', 'wb') as model_file:
    dill.dump(nn,model_file)

print("Finish!")

input("Press Enter to exit...")
