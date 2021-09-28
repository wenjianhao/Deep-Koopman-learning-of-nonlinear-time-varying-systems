import torch
import torch.nn as nn

NUM = 8
NUM_PSI_OUTPUT = NUM
NUM_A_OUTPUT = NUM
NUM_B_OUTPUT = NUM
NUM_C_OUTPUT = 6


# Build PSI network
class PSI_NN(nn.Module):

    def __init__(self):
        super(PSI_NN, self).__init__()
        # 4 hidden layers
        self.inputsize = 3
        self.hidd1size = 48
        self.hidd2size = 64
        self.hidd3size = 24
        self.outputsize = NUM_PSI_OUTPUT
        
        self.psi_hidden_layers = nn.Sequential(

            nn.Linear(self.inputsize, self.hidd1size, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidd1size, self.hidd2size, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidd2size, self.hidd3size, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidd3size, self.outputsize, bias=True)
        )

        #print(self.psi_hidden_layers)
        
    def forward(self, input):  
        input = input
        output = self.psi_hidden_layers(input)
        return output

# Build A network
class A_NN(nn.Module):

    def __init__(self):
        super(A_NN, self).__init__()
        # 1 hidden layers
        self.inputsize = NUM_A_OUTPUT
        self.hiddsize = 128
        self.hidd1size = 64
        self.hidd2size = 32
        self.outputsize = NUM_A_OUTPUT

        self.a_hidden_layer = nn.Sequential(

            nn.Linear(self.inputsize, self.hiddsize, bias=False),
            nn.Linear(self.hiddsize, self.hidd1size, bias=False),
            nn.Linear(self.hidd1size, self.hidd2size, bias=False),
            nn.Linear(self.hidd2size, self.outputsize, bias=False),
        )

        #print(self.a_hidden_layer)
        
    def forward(self, input):  
        input = input
        output = self.a_hidden_layer(input)
        return output

#Build B network
class B_NN(nn.Module):

    def __init__(self):
        super(B_NN, self).__init__()
        # 1 hidden layers
        self.inputsize = 2
        self.hiddsize = 128
        self.hidd1size = 64
        self.hidd2size = 32
        self.outputsize = NUM_B_OUTPUT

        self.b_hidden_layer = nn.Sequential(

            nn.Linear(self.inputsize, self.hiddsize, bias=False),
            nn.Linear(self.hiddsize, self.hidd1size, bias=False),
            nn.Linear(self.hidd1size, self.hidd2size, bias=False),
            nn.Linear(self.hidd2size, self.outputsize, bias=False),
        )

        #print(self.b_hidden_layer)
        
    def forward(self, input):  
        input = input
        output = self.b_hidden_layer(input)
        return output

# Build network
class C_NN(nn.Module):

    def __init__(self):
        super(C_NN, self).__init__()
        # 1 hidden layers
        self.inputsize = 6
        self.hiddsize = 1000
        self.outputsize = NUM_C_OUTPUT

        self.c_hidden_layer = nn.Sequential(

            nn.Linear(self.inputsize, self.hiddsize, bias=False),
            # nn.ELU(),
            nn.Linear(self.hiddsize, self.outputsize, bias=False),
        )

        #print(self.c_hidden_layer)
        
    def forward(self, input):  
        input = input
        output = self.c_hidden_layer(input)
        return output

# Build U_LIFT network
class U_LIFT_NN(nn.Module):

    def __init__(self):
        super(U_LIFT_NN, self).__init__()
        # 4 hidden layers
        self.inputsize = 2
        self.hidd1size = 20
        self.hidd2size = 40
        self.hidd3size = 60
        self.hidd4size = 30
        self.outputsize = 2
        
        self.UL_hidden_layers = nn.Sequential(

            nn.Linear(self.inputsize, self.hidd1size),
            nn.ELU(),
            nn.Linear(self.hidd1size, self.hidd2size),
            nn.ELU(),
            nn.Linear(self.hidd2size, self.hidd3size),
            nn.ELU(),
            nn.Linear(self.hidd3size, self.hidd4size),
            nn.ELU(),
            nn.Linear(self.hidd4size, self.outputsize)
        )

        #print(self.psi_hidden_layers)
        
    def forward(self, input):  
        input = input
        output = self.UL_hidden_layers(input)
        return output

class U_0_NN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=[16, 32, 32, 16]):
        super(U_0_NN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        # out = F.softmax(self.layers[-1](x)) # classifier
        out = self.layers[-1](x)
        return out