import torch
import torch.nn as nn
import torchvision
import numpy as np

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(128)
        self.max_pool = nn.AdaptiveMaxPool2d(128)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel % ratio, kernel_size=3, stride=1, padding=1,bias=False),
                #channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels = channel % ratio, out_channels = channel, kernel_size = 3, stride=1, padding=1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #a = self.avg_pool(x)
        #print(a.shape)
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(128)
        self.max_pool = nn.AdaptiveMaxPool2d(128)

    def forward(self, x):
        #avgout = torch.mean(x, dim=1, keepdim=True)
        #maxout, _ = torch.max(x, dim=1, keepdim=True)
        #out = avgout(maxout, _)
        avgout = self.avg_pool(x)
        maxout = self.max_pool(x)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        #out = self.conv2d(x)
        #out = self.conv2d(x)
#        out = torch.cat([avgout, maxout], dim=1)
#        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel=channel, ratio=16)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        #print(x.shape)
        #print(self.channel_attention(x).shape)
        out = self.channel_attention(x) * x
        #print(out.shape)![](../training/DESKTOP-4S4PQGI_2023-03-09_H19-35-49_256_1_1_1_1_batch_l2_0.75_1colorIn1color_main_udh/trainPics/ResultPics_epoch008_batch1999.png)
        #print('outchannels:{}'.format(out.shape))
        #print(self.spatial_attention(out).shape)
        out = self.spatial_attention(out)
        return out
'''x = torch.ones(1,3,128,128)

model = CBAM(channel=3)
for name, parameters in model.named_parameters():
    print(name,parameters.size())
y = model(x)
print(y.shape)'''