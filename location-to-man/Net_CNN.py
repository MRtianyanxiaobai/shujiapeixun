import  torch.nn as nn
class cnn_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,#输入的通道数
                    out_channels=16,#卷积核的数量
                    kernel_size=3,#视野域
                    stride=1,#步长
                    padding=2
             ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16,32,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64,128,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.mlp_layers=nn.Sequential(
            nn.Linear(128*20*20,4)
        )

    def forward(self, x):
        cnn_out = self.cnn_layers(x)
        print(cnn_out.size())
        cnn_out = cnn_out.view(-1, 128 * 20 * 20)
        output = self.mlp_layers(cnn_out)
        return output