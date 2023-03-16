import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
from PreProcessing import PreprocessData
import torch.nn as nn
import torch.optim as optim
# Set Seeds For Randomness
torch.manual_seed(10)
np.random.seed(10)    
InputSize = 6  # Input Size
batch_size = 1 # Batch Size Of Neural Network
OutputSize = 1 # Output Size 

import matplotlib.pyplot as plt

def my_plot(epochs, loss):
    plt.plot(epochs, loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
############################################# Network #####################################

NumEpochs = 25
HiddenSize = 500

# Create The Neural Network Model
class Net(nn.Module):
    def __init__(self, InputSize,OutputSize):
        super(Net, self).__init__()
		###### Define The Feed Forward Layers Here! ######
        self.fc1 = nn.Linear(InputSize, HiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HiddenSize, OutputSize)
        
    def forward(self, x):
		###### Write Steps For Forward Pass Here! ######
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(InputSize, OutputSize)     

criterion = nn.MSELoss() ###### Define The Loss Function Here! ######
optimizer = optim.Adam(net.parameters(), lr=0.001) ###### Define The Optimizer Here! ######

##################################################################################################

if __name__ == "__main__":
    
    loss_vals =  []    
    TrainSize,SensorNNData,SensorNNLabels = PreprocessData()   
    for j in range(NumEpochs):
        losses = 0
        epoch_loss= []
        for i in range(TrainSize):  
            input_values = Variable(SensorNNData[i])
            labels = Variable(SensorNNLabels[i])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            losses += loss.item()
        loss_vals.append(sum(epoch_loss)/len(epoch_loss))
            
        print ('Epoch %d, Loss: %.4f' %(j+1, losses/SensorNNData.shape[0]))       
        torch.save(net.state_dict(), './SavedNets/NNBot.pkl')
           
        


    my_plot(np.linspace(1, NumEpochs, NumEpochs).astype(int), loss_vals)