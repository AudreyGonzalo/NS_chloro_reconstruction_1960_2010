import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_net(torch.nn.Module):

    def __init__(self, params):
        super(MLP_net,self).__init__()
        self.inputLayer   = torch.nn.Linear(params['dim_input'], params['dim_hidden_1'])
        self.hiddenLayer1 = torch.nn.Linear(params['dim_hidden_1'], params['dim_hidden_2'])
        self.hiddenLayer2 = torch.nn.Linear(params['dim_hidden_2'], params['dim_hidden_3'])
        self.hiddenLayer3 = torch.nn.Linear(params['dim_hidden_3'], params['dim_output'])

    def forward(self, inp):
        res = self.inputLayer(inp)
        res = self.hiddenLayer1(res)
        res = self.hiddenLayer2(res)
        res = self.hiddenLayer3(res)
        return res

class INT_net(torch.nn.Module):

    def __init__(self, params):
        super(INT_net, self).__init__()
        self.Dyn_net = MLP_net(params)

    def forward(self, inp, dt):
        k1     = self.Dyn_net(inp)
        inp_k2 = inp + 0.5*dt*k1
        k2     = self.Dyn_net(inp_k2)
        inp_k3 = inp + 0.5*dt*k2
        k3     = self.Dyn_net(inp_k3)
        inp_k4 = inp + dt*k3
        k4     = self.Dyn_net(inp_k4)
        pred   = inp +dt*(k1+2*k2+2*k3+k4)/6
        return pred
