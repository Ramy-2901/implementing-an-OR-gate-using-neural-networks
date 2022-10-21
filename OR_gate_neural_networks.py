# -*- coding: utf-8 -*-
"""
@author: ramya
"""
#OR gate simulator 
#For an OR gate we want the output to be close to 1 when one or more of the inputs is 1 and 0 otherwise.
# Importing the libraries
import numpy as np
import random
import math
class node:
    def __init__(self,weights_list):
        import random
        self.weights=[random.random() for i in range(weights_list+1)]
        self.output = 0
        self.inputs =[]
#input is the values randomly being generated 
    def sum_weights(self):
        sum = self.weights[0]
        for i in range(1,len(self.weights)-1):
            sum += self.weights[i] + self.inputs[i-1]
        return sum
#The reason we use sigmoid function is because it exists between (0 to 1). It is especially used for models where we have to predict the probability as an output.
def sigmoid(x):
    import numpy as np
    return 1/(1+np.exp(-x))

def build_network(no_layers,layer_array,no_of_inputs):
    network = []
    network.append([node(no_of_inputs+1) for i in range(layer_array[0])])
    for i in range(1,no_layers-1):
        network.append([node(layer_array[i-1]) for j in range(layer_array[i])])
    network.append([node(layer_array[no_layers-1]) for i in range(layer_array[no_layers-1])])
    return network

def make_data(data_size = 100):
    import random
    i = []
    for j in range(data_size):
        t = [random.randint(0,2),random.randint(0,2)]
        if t[0] ==1 or t[1]==1:
            t.append(1)
        else:
            t.append(0)
        i.append(t)
    return i

def forward_propogate(network,inputs):
    for i in network[0]:
        i.inputs += inputs
        i.output = sigmoid(i.sum_weights())

    for i in range(1,len(network)):
        for j in network[i]:
            j.inputs += [k.output for k in network[i-1]]
            j.output = sigmoid(j.sum_weights())
    return network

def error(network,layers,expected):
    e = []
    for i in range(len(network[layers-1])):
        e.append(expected-network[layers-1][i].output)
    return e


def back_propogate(network, no_layers,expected):
    e = error(network,no_layers,expected)
    for i in range(len(network[no_layers-1])):
        e[i] *= (network[no_layers-1][i].output)*(1-network[no_layers-1][i].output)
    
    for i in range(no_layers-1,-1,-1):
        e2 = []
        for j in range(len(network[i])):
            l = network[i][j].weights
            for k in range(len(network[i][j].weights)):
                if(k == 0):
                    network[i][j].weights += e[j]
                else:
                    if(k == 2):
                        print(network[i][j].weights)
                    network[i][j].weights[k] += e[j]*network[i][j].inputs[k-1]
                if(j == 0):
                    e2.append(l[k]*e[j])
                else:
                    if k !=0:
                        a = l[k]
                        s = e[j]
                        e2[k]+=a*s*(network[i][j].inputs[k-1])*(1-network[i][j].inputs[k-1])
        e = e2
    return network

def clear(network):
    for i in network:
        for j in i:
            j.inputs = []
            j.output = 0
    return network

l = int(input("Enter training size: "))
data = make_data(l)
network = build_network(3,[2,2,2],2)

for i in range(int(l/4)):
    network = forward_propogate(network, [data[i][0],data[i][1]])
    network = back_propogate(network, 3, data[i][2])
    network = clear(network)
    
t = 0
for i in range(int(l/4),l):
    if(network[4][0] > network[4][1] and data[i][2] == 0):
        t+=1

print(t*4/l)
