from fastai.layers import *
from fastai.torch_core import *
import torch
import torch.nn as nn
import torchvision.models as models
# The code below is meant to be merged into fastaiv1 ideally


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: Optional[NormType] = NormType.Batch,
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size #5 for 1024 channels and 5 for 2048
    """

    def __init__(self, channel, k_size=5):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class Res18Feature(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0):
        super().__init__()
        self.drop_rate = drop_rate

        resnet = models.resnet101(pretrained=False)
        state_dict = torch.load('/root/autodl-tmp/DeOldify-master/models/resnet101-63fe2227.pth')
        resnet.load_state_dict(state_dict)

        self.features = nn.Sequential(*list(resnet.children())[:-2])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        # self.attention = nn.Sequential(
        #     nn.Conv2d(fc_in_dim, fc_in_dim, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(fc_in_dim),
        #     nn.BatchNorm2d(fc_in_dim),
        #     nn.Sigmoid(),
        # )

        self.attention = LKAttention(fc_in_dim)

        # self.SpatialGate = SpatialGate()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.alpha = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.eca = eca_layer( channel = 512, k_size = 5)

    def forward(self, x,emb):
        x = self.features(x) #([64, 512, 7, 7])
        # print(x.shape)
        global_attention_mask = self.attention(x)
        global_features = global_attention_mask * x

        # global_features = self.SpatialGate(x)

        outx = self.avg_pool(global_features)
        outx = nn.Dropout(self.drop_rate)(outx)
        outx = outx.view(outx.size(0), -1)
        output = self.fc(outx)

        x = self.eca(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        # attention_weights = self.alpha(x)
        # out = attention_weights * self.fc(x)

        return global_features

import torch.nn.functional as F
from torch.nn import Softmax

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B*W, 1, 1)

class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Moudle"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax    = Softmax(dim=3)
        self.INF        = INF
        self.gamma      = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x = x.cuda()
        m_batchsize, _, height, width = x.size()
        
        proj_query = self.query_conv(x)
        # b, c', h, w ===> b, w, c', h ===> b*w, c', h ===> b*w, h, c'
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height).permute(0, 2, 1)
        # b, c', h, w ===> b, h, c', w ===> b*h, c', w ===> b*h, w, c'
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x)
        # b, c', h, w ===> b, w, c', h ===> b*w, c', h
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        # b, c', h, w ===> b, h, c', w ===> b*h, c', w
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)
        
        proj_value = self.value_conv(x)
        # b, c', h, w ===> b, w, c', h ===> b*w, c', h
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        # b, c', h, w ===> b, h, c', w ===> b*h, c', w
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)
        
        # torch.bmm((b*w,h,c')x(b*w,c',h))===>(b*w,h,h)+(b*w,h,h)===>(b*w,h,h)===>(b,w,h,h)===>(b, h, w, h)

        # print(proj_query_H.is_cuda,proj_key_H.is_cuda,x.is_cuda)
        if x.is_cuda:
            energy_H = (torch.bmm(proj_query_H, proj_key_H).cuda()+self.INF(m_batchsize, height, width).cuda()).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        else: 
            energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)

        # torch.bmm((b*h,w,c')x(b*h,c',w))===>(b*h,w,w)===>(b, h, w, w)
        if x.is_cuda:
            energy_W = (torch.bmm(proj_query_W, proj_key_W).cuda()).view(m_batchsize, height, width, width)
        else: 
            energy_W = (torch.bmm(proj_query_W, proj_key_W)).view(m_batchsize, height, width, width)

        # torch.cat([(b,h,w,h),(b,h,w,w)], 3)===>(b,h,w,h+w)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        
        # (b,h,w,h+w)===>(b,h,w,h)===>(b,w,h,h)===>(b*w,h,h)
        att_H = concate[:,:,:,0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize*width, height, height)
        # (b,h,w,h+w)===>(b,h,w,w)===>(b*h,w,w)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height, width, width)
        
        # torch.bmm((b*w,c',h)x(b*w,h,h))===>(b*w,c',h)===>(b,w,c',h)===>(b,c',h,w)
        if x.is_cuda:
            out_H = torch.bmm(proj_value_H.cuda(), att_H.permute(0, 2, 1).cuda()).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        else: 
            out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)

        # torch.bmm((b*h,c',w)x(b*h,w,w))===>(b*h,c',w)===>(b,h,c',w)===>(b,c',h,w)
        if x.is_cuda:
            out_W = torch.bmm(proj_value_W.cuda(), att_W.permute(0, 2, 1).cuda()).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        else: 
            out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        # out_H = out_H.cpu()
        # out_W = out_H.cpu()
        # print(out_H.is_cuda,out_W.is_cuda)
        if x.is_cuda:
            out_final = self.gamma.cuda()*(out_H + out_W)
        else: 
            out_final = self.gamma*(out_H + out_W)

        return out_final + x