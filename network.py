import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ContemptNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=50)
        self.relu1 = nn.ReLU()
        # TODO: batch norm?
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=50, out_features=100)
        self.prelu2 = nn.PReLU() # parametric relu - if x < 0, returns 0.25*x - o.w. return x

        self.fc3 = nn.Linear(in_features=100, out_features=160)
        self.prelu3 = nn.PReLU()

        self.out = nn.Linear(in_features=160, out_features=3)
        self.out_act = nn.Softmax()

    def forward(self, input):
        layer1 = self.fc1(input)
        layer1_act = self.relu1(layer1)

        layer2 = self.fc2(layer1)
        layer2_act = self.prelu2(layer2)
        layer2_dout = self.dout(layer2_act)

        # layer2_out = self.out(layer2_act)

        layer3 = self.fc3(layer2_dout)
        layer3_act = self.prelu3(layer3)
        layer3_out = self.out(layer3_act)

        output_classes = self.out_act(layer3_out)
        return output_classes
        

m = nn.PReLU()
input = torch.randn(2)
output = m(input)
print(input)
print(output)