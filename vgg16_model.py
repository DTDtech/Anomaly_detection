import torch

class VGG16(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=3,
                            out_channels=64,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2),
                            stride=(2, 2, 2))            
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=64,
                            out_channels=128,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=128,
                            out_channels=128,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2),
                            stride=(2, 2, 2))        
        )

        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=128,
                            out_channels=256,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=256,
                            out_channels=256,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=256,
                            out_channels=256,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2),
                            stride=(2, 2, 2))        
        )

        self.block_4 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=256,
                            out_channels=512,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2),
                            stride=(2, 2, 2))
        )

        self.block_5 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.Conv3d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.ReLU(), 
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2),
                            stride=(2, 2, 2))
        )

        for m in self.modules():
            if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x


