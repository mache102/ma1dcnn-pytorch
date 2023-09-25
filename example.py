import torch

from torchsummary import summary

from ma1dcnn import MA1DCNN

bsz = 32
x = torch.randn(bsz, 1, 2048)
model = MA1DCNN(num_classes=12, in_channels=x.size(1))

y = model(x)
print(y.shape)

# summary option
model = MA1DCNN(num_classes=12, in_channels=1)
summary(model, (1, 4096))
