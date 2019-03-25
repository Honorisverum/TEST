import torch.nn as nn
import torch
import torchvision

"""
=================================================
                    CREATE CNN
=================================================
"""


class PretrainedCNN(nn.Module):

    def __init__(self, img_dim, out_dim):
        super(PretrainedCNN, self).__init__()

        assert img_dim == 224, "incorrect image size as input to ResNet"

        # input image size: (224, 224)
        self.net = torchvision.models.resnet50(pretrained=True)  # 18, 34, 50

        n_features = self.net.fc.in_features
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        return self.net(x)


class CustomCNN(nn.Module):
    def __init__(self, img_dim, out_dim):
        super(CustomCNN, self).__init__()

        self.aux_dim = img_dim
        self.out_dim = out_dim

        # convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.aux_dim -= 4

        # pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.aux_dim //= 2

        # convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.aux_dim -= 4

        # polling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.aux_dim //= 2

        # convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.aux_dim -= 2

        # pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.aux_dim //= 2

        # fully connected layer
        self.fc = nn.Linear(64 * self.aux_dim ** 2, self.out_dim)

    def forward(self, x):

        # torch(batch_size, 32, D, D)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # torch(batch_size, 32 * D * D)
        x = x.view(x.size(0), -1)

        return self.fc(x)


if __name__ == "__main__":
    " some test's "

    img_size = 224
    out_dim = 100
    batch_size = 1
    pretrained_net = PretrainedCNN(img_size, out_dim)
    ex = torch.randn(batch_size, 3, img_size, img_size)
    ans = pretrained_net(ex)
    assert ans.size() == torch.Size([batch_size, out_dim]), f"incorrect out size for PretrainedCNN: {ans.size()}"
    print("PretrainedCNN passed test!")

    img_size = 224
    out_dim = 100
    batch_size = 1
    custom_net = CustomCNN(img_size, out_dim)
    ex = torch.randn(batch_size, 3, img_size, img_size)
    ans = custom_net(ex)
    assert ans.size() == torch.Size([batch_size, out_dim]), f"incorrect out size for CustomCNN: {ans.size()}"
    print("CustomCNN passed test!")
