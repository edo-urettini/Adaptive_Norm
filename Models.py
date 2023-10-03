import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

#Define a neural network class that takes the past 200 time steps as input and outputs the next 50 time steps. Use pytorch.
class Net(nn.Module):
    def __init__(self): #Define the layers
        super(Net, self).__init__()
        self.relu=nn.ReLU()     
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
    
    def encoder(self, x):     
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
        

    def forward(self, x): #Define the forward pass        
        x=self.encoder(x)
        x = self.fc3(x)
        return x
    


class GAS_Net(nn.Module):
    def __init__(self):
        super(GAS_Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        #self.output = nn.Linear(250, 50, bias=False)
        self.sigma_layer = nn.Linear(200, 50)
        self.mu_layer = nn.Linear(200, 50)
    
    def encoder(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    
        
    def forward(self, x, mu_vector, sigma2_vector, phase):
        
        #In phase 1 we use only the mu_layer by setting the encoded x to zero
        if phase == 0:
            mu_encoded = self.mu_layer(mu_vector)
            output = torch.add(mu_encoded, 0)
        
        #In phase 2 we use both the mu_layer and the encoding of x but we set the encoded sigma to one
        elif phase == 1:
            encoded = self.encoder(x)
            mu_encoded = self.mu_layer(mu_vector)
            output = torch.add(encoded, mu_encoded)

        #In phase 3 we use everything

        elif phase == 2:
            encoded = self.encoder(x)
            #Multiply the encoded x with an new encoding of the standard deviation
            sigma_vector= torch.sqrt(sigma2_vector)        
            sigma_encoded = self.sigma_layer(sigma_vector)
            encoded = torch.mul(encoded, sigma_encoded)
            #Sum the encoded x with an new encoding of the mean
            mu_encoded = self.mu_layer(mu_vector)
            output = torch.add(encoded, mu_encoded)
            
        return output



#REVIN net

from RevIN import RevIN

class Revin_Net(nn.Module):
    def __init__(self): #Define the layers
        super(Revin_Net, self).__init__()
        self.relu=nn.ReLU()     
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.revin_layer = RevIN(1)
    
    def encoder(self, x):     
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
        

    def forward(self, x): #Define the forward pass
        x = x.unsqueeze(0).unsqueeze(2)
        x = self.revin_layer(x, 'norm')
        x = x.squeeze(2).squeeze(0)
        x = self.encoder(x)
        x = self.fc3(x)
        x = x.unsqueeze(0).unsqueeze(2)
        x = self.revin_layer(x, 'denorm')
        x = x.squeeze(2).squeeze(0)
        return x