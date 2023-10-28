from cnn_functions import dlFunctions
import numpy as np


functions = dlFunctions()

with open('hw3testfile.txt', 'r') as f:
    content = []
    for line in f.readlines():
        line = line.strip()
        content.append(line)

for i in range(len(content)):
    if content[i][0] == 'f' or content[i][0] == 'b':
        num_inputs = int(content[i+1])
        inputs = []
        for j in range(i+2, i+2+num_inputs):
            temp = content[j].split(' ')
            dim1 = int(temp[0])
            dim2 = int(temp[1])           
            inputs.append(np.array(temp[2:],dtype='float').reshape(dim1, dim2, order='F'))
            
        if content[i] == 'forw_relu':
            print('forw_relu')
            print(functions.forw_relu(inputs[0]))
    
        elif content[i] == 'forw_maxpool':
            print('forw_maxpool')
            print(functions.forw_maxpool(inputs[0]))
        
        elif content[i] == 'forw_meanpool':
            print('forw_meanpool')
            print(functions.forw_meanpool(inputs[0]))
        
        elif content[i] == 'forw_fc':
            print('forw_fc')
            print(functions.forw_fc(inputs[0], inputs[1], inputs[2]))
        
        elif content[i] == 'forw_softmax':
            print('forw_softmax')
            print(functions.forw_softmax(inputs[0]))
        
        elif content[i] == 'back_relu':
            print('back_relu')
            print(functions.back_relu(inputs[0], inputs[1], inputs[2]))
    
        elif content[i] == 'back_maxpool':
            print('back_maxpool')
            print(functions.back_maxpool(inputs[0], inputs[1], inputs[2]))
        
        elif content[i] == 'back_meanpool':
            print('back_meanpool')
            print(functions.back_meanpool(inputs[0], inputs[1], inputs[2]))
        
        elif content[i] == 'back_fc':
            print('back_fc')
            dzdx, dzdw, dzdb = functions.back_fc(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
            print(dzdx, dzdw, dzdb)
            
        elif content[i] == 'back_softmax':
            print('back_softmax')
            print(functions.back_softmax(inputs[0], inputs[1], inputs[2]))
    else:
        continue
