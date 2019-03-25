import torch.nn as nn
import torch
import torchvision

"""
=================================================
        CREATE NETWORK CNN + LSTM
=================================================
"""


class PretrainedCNN(nn.Module):
    def __init__(self, out_dim):
        super(PretrainedCNN, self).__init__()

        # input image size: (224,224)
        self.net = torchvision.models.resnet50(pretrained=True)  # 18, 34, 50
        
        n_features = self.net.fc.in_features
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.fc = nn.Linear(n_features, out_dim)
    
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    " some test's "
    net = PretrainedCNN(100)
    ex = torch.randn(1, 3, 224, 224)
    ans = net(ex)
    print(ans.size())
