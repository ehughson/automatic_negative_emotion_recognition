import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ContemptNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=17, out_features=50)
        self.relu1 = nn.ReLU()
        # TODO: batch norm?
        self.dout = nn.dropout(0.2)
        self.fc2 = nn.Linear(in_features=50, out_features=100)
        self.prelu = nn.PReLU() # parametric relu - if x < 0, returns 0.25*x - o.w. return x
        self.out = nn.Linear(in_features=100, out_features=3)
        self.out_act = nn.Softmax()

    def forward(self, input):
        layer1 = self.fc1(input)
        layer1_act = self.relu1(layer1)
        layer1_dout = self.dout(layer1_act)

        layer2 = self.fc2(layer1_dout)
        layer2_act = self.prelu(layer2)
        layer2_out = self.out(layer2_act)
        output_classes = self.out_act(layer2_out)
        return output_classes
        

m = nn.PReLU()
input = torch.randn(2)
output = m(input)
print(input)
print(output)