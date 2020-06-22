from resnet_setex import *
from resnet_setex import BasicBlock, Bottleneck, conv1x1, conv3x3
import torch
import torch.nn as nn

class Resnet_Expl(torch.nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], num_classes=312):
        super(Resnet_Expl, self).__init__()
        # self.main_model = main_model

        self.num_classes = num_classes
        self.layers = layers
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.replace_stride_with_dilation = [False, False, False]
        self.block = Bottleneck
        self.zero_init_residual = False

        # self.normalize_x0 = nn.BatchNorm2d(64)
        # self.normalize_x1 = nn.BatchNorm2d(256)
        # self.normalize_x2 = nn.BatchNorm2d(512)
        # self.normalize_x3 = nn.BatchNorm2d(1024)
        # self.normalize_x4 = nn.BatchNorm2d(2048)

        self.layer1 = self._make_layer(self.block, 64, layers[0])
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2,
                                       dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2,
                                       dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2,
                                       dilate=self.replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.expl_fc = nn.Linear(512 * self.block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        # torch.Size([8, 64, 56, 56]) torch.Size([8, 256, 56, 56]) torch.Size([8, 512, 28, 28]) torch.Size([8, 1024, 14, 14]) torch.Size([8, 2048, 7, 7])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x0, x1, x2, x3, x4):
        # main_predictions, x0, x1, x2, x3, x4 = main_model(x)
        # return main_predictions, x0, x1, x2, x3, x4
        
        # x0 = self.normalize_x0(x0)
        # x1 = self.normalize_x1(x1)
        # x2 = self.normalize_x2(x2)
        # x3 = self.normalize_x3(x3)
        # x4 = self.normalize_x4(x4)

        x1_own = self.layer1(x0)
        # x1_own = torch.cat((x1_own, x1), dim=1)
        x1_own = x1_own.add(x1)

        x2_own = self.layer2(x1_own)
        x2_own = x2_own.add(x2)

        x3_own = self.layer3(x2_own)
        x3_own = x3_own.add(x3)
        # print(x3[0])
        # print(x3_own[0])

        x4_own = self.layer4(x3_own)
        x4_own = x4_own.add(x4)
        # print(x4[0])
        # print(x4_own[0])

        expl_predictions = self.avgpool(x4_own)
        expl_predictions = torch.flatten(expl_predictions, 1)
        # expl_predictions = expl_predictions.view(expl_predictions.size(0), -1)
        expl_predictions  = self.expl_fc(expl_predictions)

        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape, main_predictions.shape)
        # print(x0.shape, x1_own.shape, x2_own.shape, x3_own.shape, x4_own.shape, expl_predictions.shape)
        return expl_predictions