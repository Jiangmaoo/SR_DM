
import torch
from torch import nn
import torch.nn.functional as F
from models.layers import BasicConv, ResLayers, FAM, SCM


class EBlock(nn.Module):
    def __init__(self, mode, num_res):
        super(EBlock, self).__init__()
        base_channel = 32
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 8, base_channel * 8, kernel_size=3, relu=True, stride=1),
        ])
        self.res = nn.ModuleList([
            ResLayers(base_channel, num_res, mode),
            ResLayers(base_channel * 2, num_res, mode),
            ResLayers(base_channel * 4, num_res, mode),
            ResLayers(base_channel * 8, num_res, mode)
        ])
        self.FAM3 = FAM(base_channel * 8)
        self.FAM2 = FAM(base_channel * 4)
        self.FAM1 = FAM(base_channel * 2)
        self.FAM0 = FAM(base_channel)
        self.SCM3 = SCM(base_channel * 8)
        self.SCM2 = SCM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_8 = F.interpolate(x_4, scale_factor=0.5)
        z2 = self.SCM1(x_2)
        z4 = self.SCM2(x_4)
        z8 = self.SCM3(x_8)

        # 256*256
        x_ = self.feat_extract[0](x)
        res1 = self.res[0](x_)
        res1 = self.FAM0(x_, res1)
        # 128*128
        z = self.feat_extract[1](res1)
        # z = self.FAM1(z, z2)
        res2 = self.res[1](z)
        res2 = self.FAM1(z, res2)
        # 64*64
        z = self.feat_extract[2](res2)
        # z = self.FAM2(z, z4)
        res3 = self.res[2](z)
        res3 = self.FAM2(z, res3)

        z = self.feat_extract[3](res3)
        # z = self.FAM3(z, z8)
        res4 = self.res[3](z)
        z = self.FAM3(z, res4)

        z = self.feat_extract[4](z)
        z = self.res[3](z)

        feature_dic = {
            'res1': res1,  # [2, 32, 128, 128])
            'res2': res2,  # [2, 64, 64, 64])
            'res3': res3,  # [2, 128, 32, 32])
            'z': z,        # [2, 256, 8, 8])
            'x': x,        # [2, 3, 128, 128])
            'x_2': x_2,    # [2, 3, 64, 64])
            'x_4': x_4,    # [2, 3, 32, 32])
            'x_8': x_8     # [2, 3, 16, 16])
        }
        return feature_dic


class DBlock(nn.Module):
    def __init__(self, mode, num_res):
        super(DBlock, self).__init__()
        base_channel = 32
        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 8, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )
        self.res = nn.ModuleList([
            ResLayers(base_channel, num_res, mode),
            ResLayers(base_channel * 2, num_res, mode),
            ResLayers(base_channel * 4, num_res, mode),
            ResLayers(base_channel * 8, num_res, mode)
        ])
        self.FAM3 = FAM(base_channel * 8)
        self.FAM2 = FAM(base_channel * 4)
        self.FAM1 = FAM(base_channel * 2)
        self.FAM0 = FAM(base_channel)

    def forward(self, x, x0):
        outputs = list()

        # -------------32*8------------------------
        z = self.FAM3(x['z'], x0['z'])
        z = self.res[3](z)
        z_ = self.ConvsOut[0](z)   # 32*8->3

        z = self.feat_extract[0](z)
        # outputs.append(z_ + x['x_8'])

        # -------------32*4------------------------
        res3 = self.FAM2(x['res3'], x0['res3'])
        # print("===============")
        # print(z.shape)    # [2,128,16,16]
        # print(res3.size())  # [2,128,32,32]
        z = torch.cat([z, res3], dim=1)
        z = self.Convs[0](z)
        z = self.res[2](z)
        z_ = self.ConvsOut[1](z)  # 32*4->3
        z = self.feat_extract[1](z)
        outputs.append(z_ + x['x_4'])

        # ---------------32*2----------------------
        res2 = self.FAM1(x['res2'], x0['res2'])
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[1](z)
        z = self.res[1](z)
        z_ = self.ConvsOut[2](z)  # 32*2->3
        # 256*256
        z = self.feat_extract[2](z)
        outputs.append(z_ + x['x_2'])

        # --------------32-----------------------
        res1 = self.FAM0(x['res1'], x0['res1'])
        # res1 = torch.cat([x['res1'] + x0['res1']], dim=1)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[2](z)
        z = self.res[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z + x['x'])

        return outputs


class SFNet(nn.Module):
    def __init__(self, mode, num_res=4):
        super(SFNet, self).__init__()

        self.encoder_gt = EBlock(mode, num_res)
        self.encoder_x = EBlock(mode, num_res)
        self.encoder_general = EBlock(mode, num_res)

        self.decoder_noise = DBlock(mode, num_res)
        self.decoder_gt = DBlock(mode, num_res)

    def forward(self, x, gt):
        x_feature = self.encoder_x(x)
        gt_feature = self.encoder_gt(x)

        x0_feature = self.encoder_general(x)
        gt0_feature = self.encoder_general(gt)

        output_n = self.decoder_noise(x_feature, x0_feature)
        output_gt = self.decoder_gt(gt_feature, gt0_feature)

        return output_n, output_gt


def build_net(mode):
    return SFNet(mode)

if __name__ == '__main__':
    size = (3, 3, 256, 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input1 = torch.ones(size).to(device)
    input2 = torch.ones(size).to(device)


    model = build_net('train')
    model.to(device)
    output_n, output_gt = model(input1, input1)