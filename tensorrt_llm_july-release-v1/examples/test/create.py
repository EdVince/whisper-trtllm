import torch
import torch.nn as nn

class SimpleConvTorchNet(nn.Module):
    def __init__(self):
        super(SimpleConvTorchNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

if __name__ == '__main__':

    torch_net = SimpleConvTorchNet()
    torch.save(torch_net.state_dict(),'weight.pth')