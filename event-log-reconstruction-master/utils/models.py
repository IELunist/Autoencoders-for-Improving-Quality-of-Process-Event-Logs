import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms

class VAE(nn.Module):
    # This is a method proposed by https://arxiv.org/pdf/1312.6114.pdf
    # x --> fc1 --> relu --> fc2 --> z --> fc3 --> relu --> fc4 --> sigmoid --> x'
    def __init__(self, shape, layer1, layer2, isCuda):
        '''
        input size: (batch, sequence_length, feature)
        shape: tuple (shape[1], shape[2])/(sequence_length, feature) of c_train
        layer1: dim of hidden layer1
        layer2: dim of hidden layer2
        '''
        super(VAE, self).__init__()

        self.shape = shape
        self.isCuda = isCuda
        
        self.fc1 = nn.Linear(shape[1]*shape[2], layer1) #encode
        self.fc21 = nn.Linear(layer1, layer2) #encode
        self.fc22 = nn.Linear(layer1, layer2) #encode
        self.fc3 = nn.Linear(layer2, layer1) #decode
        self.fc4 = nn.Linear(layer1, shape[1]*shape[2]) #decode
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        if self.isCuda:
            self.cuda()

        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc21.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc22.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        #x --> fc1 --> relu --> fc21
        #x --> fc1 --> relu --> fc22
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2)
        if self.isCuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps*std + mu

    def decode(self, z, x):
        #z --> fc3 --> relu --> fc4 --> sigmoid
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3)).view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        #mu= mean
        #logvar = log variational
        mu, logvar = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar
    
class VAE_dropout(nn.Module):
    # Apply Dropout to both input and hidden layers
    # x --> dropout --> fc1 --> relu --> dropout --> fc2 --> z --> dropout --> fc3 --> relu --> dropout -->fc4 --> sigmoid --> x'
    def __init__(self, shape, layer1, layer2, isCuda):
        '''
        shape: tuple (shape[1], shape[2]) of c_train
        layer1: dim of hidden layer1
        layer2: dim of hidden layer2
        
        '''
        super(VAE_dropout, self).__init__()

        self.shape = shape
        self.isCuda = isCuda
        
        self.fc1 = nn.Linear(shape[1]*shape[2], layer1) #encode
        self.fc21 = nn.Linear(layer1, layer2) #encode
        self.fc22 = nn.Linear(layer1, layer2) #encode

        self.fc3 = nn.Linear(layer2, layer1) #decode
        self.fc4 = nn.Linear(layer1, shape[1]*shape[2]) #decode

        
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dout = nn.Dropout(p=0.2)
        
        if self.isCuda:
            self.cuda()

        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc21.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc22.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> dropout --> fc1 --> relu --> dropout --> fc2
        # x --> dropout --> fc1 --> relu --> dropout --> fc21
        # x --> dropout --> fc1 --> relu --> dropout --> fc22
        dx = self.dout(x)
        h1 = self.relu(self.fc1(dx))
        h1 = self.dout(h1)
        return self.fc21(h1), self.fc22(h1)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2)
        if self.isCuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps*std + mu

    def decode(self, z, x):
        # z --> dropout --> fc3 --> relu --> dropout -->fc4 --> sigmoid--> x'
        dz = self.dout(z)
        h3 = self.relu(self.fc3(dz))
        h3 = self.dout(h3)
        return self.sigmoid(self.fc4(h3)).view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        #mu= mean
        #logvar = log variational
        mu, logvar = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar    
      
class AE(nn.Module):
    # x --> fc1 --> fc2 --> sigmoid --> z --> fc3 --> fc4 --> sigmoid --> x'
    def __init__(self, shape, h_dim, z_dim):
        super(AE, self).__init__()
        self.shape = shape
        
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #decode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode
    
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> fc1 --> fc2 --> sigmoid --> z
        h = self.fc1(x)
        z = self.sigmoid(self.fc2(h))
        return z

    def decode(self, z, x):
        # z --> fc3 --> fc4 --> sigmoid --> x'
        h = self.fc3(z)
        recon_x = self.sigmoid(self.fc4(h))
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)
    
class AE_tanh(nn.Module):
    # x --> fc1 --> tanh --> fc2 --> tanh --> z --> fc3 --> tanh --> fc4 --> sigmoid --> x'
    def __init__(self, shape, h_dim, z_dim):
        super(AE_tanh, self).__init__()
        self.shape = shape
        
        # X --> fc1 --> Z --> fc2 --> X'
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #decode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode
    
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> fc1 --> tanh --> fc2 --> tanh --> z
        h = self.tanh(self.fc1(x))
        z = self.tanh(self.fc2(h))
        return z

    def decode(self, z, x):
        # z --> fc3 --> tanh --> fc4 --> sigmoid --> x'
        h = self.tanh(self.fc3(z))
        recon_x = self.sigmoid(self.fc4(h))
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)
    
class AE_dropout(nn.Module):
    # x --> dropout --> fc1 --> tanh ---> dropout --> fc2 --> z --> dropout --> fc3 --> tanh --> dropout --> fc4 --> sigmoid --> x'
    def __init__(self, shape, h_dim, z_dim):
        super(AE_dropout, self).__init__()
        self.shape = shape
        
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #encode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode

        self.dout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_normal(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc2.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> dropout --> fc1 --> tanh ---> dropout --> fc2 --> z
        dx = self.dout(x)
        h = self.tanh(self.fc1(dx))
        h = self.dout(h)
        z = self.fc2(h)
        return z

    def decode(self, z, x):
        # z --> dropout --> fc3 --> tanh --> dropout --> fc4 --> sigmoid --> x'
        dz = self.dout(z)
        h = self.tanh(self.fc3(dz))
        h = self.dout(h)
        recon_x = self.sigmoid(self.fc4(h))
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        #initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size).zero_(), requires_grad=False)
        c0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size).zero_(), requires_grad=False)
        encoded_input, hidden = self.lstm(input, (h0, c0))
        return encoded_input, hidden

      
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.isCuda = isCuda
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.linear.weight, gain=np.sqrt(2))
        
    def forward(self, encoded_input, hidden):
        tt = torch.cuda if self.isCuda else torch
        decoded_output, _ = self.lstm(encoded_input, hidden)
        decoded_output = self.linear(decoded_output)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output

class LSTMAE(nn.Module):
    # x --> lstm --> z --> lstm --> fc --> sigmoid --> x'
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)
        
    def forward(self, input):
        encoded_input, hidden = self.encoder(input)
        decoded_output = self.decoder(encoded_input, hidden)
        return decoded_output