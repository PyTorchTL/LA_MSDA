import torch.nn as nn
import math
from . import multi_sources_LLMMD as LLMMD
import torch.nn.functional as F
import torch
import numpy as np

# It provides the network structure of LA-MSDA,
# including stage1 (feature extraction), Stage 2 (local label-based alignment), and stage 3 (global optimization).

# stage1 (feature extraction)
# Domain-specific
class ComEEGNet(nn.Module):

    def __init__(self):
        super(ComEEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 61), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 2))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        self.fc1 = nn.Linear(4 * 2 * 7, 1)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        # x = x.view(-1, 4 * 2 * 7)
        # x = F.sigmoid(self.fc1(x))
        return x



class DSCNN(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DSCNN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=2, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=2, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

# model framework

class LAMSDA(nn.Module):

    def __init__(self, num_classes=2, num_source=14):
        super(LAMSDA, self).__init__()

        self.sharedNet = eegnet(False)

        self.num_classes = num_classes
        self.num_source = num_source

        self.specificNet = nn.ModuleList([DSCNN(4, 256) for i in range(self.num_source)])

        self.cls_List_spec = nn.ModuleList([nn.Linear(256, self.num_classes) for i in range(self.num_source)])

        self.avgpool = nn.AvgPool2d(2, stride=1)


    def forward(self, data_src, data_tgt=0, label_src=0, mark=1, mmd_type='llmmd'):

        local_loss = 0

        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt_spec = []
            for i in range(self.num_source):
                temp = self.avgpool(self.specificNet[i](data_tgt))
                data_tgt_spec.append(temp.view(temp.size(0), -1))

            data_src = self.avgpool(self.specificNet[mark](data_src))
            data_src = data_src.view(data_src.size(0), -1)

            glo_loss = 0
            glo_loss_list = torch.empty([])

            for i in range(self.num_source):
                if i != mark:
                    glo_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_spec[mark], dim=1)
                                                   - torch.nn.functional.softmax(data_tgt_spec[i], dim=1)))
                    if glo_loss_list.shape != torch.Size([]):
                        torch.cat((glo_loss_list, torch.abs(torch.nn.functional.softmax(data_tgt_spec[mark], dim=1)
                                                          - torch.nn.functional.softmax(data_tgt_spec[i], dim=1))), 0)
                    else:
                        glo_loss_list = torch.abs(torch.nn.functional.softmax(data_tgt_spec[mark], dim=1)
                                                - torch.nn.functional.softmax(data_tgt_spec[i], dim=1))

            ## weight(global_category optimization)
            loss_weight = glo_loss_list / glo_loss
            loss_weight, _ = torch.sort(loss_weight)
            glo_loss_list, _ = torch.sort(glo_loss_list, descending=True)
            glo_loss = torch.sum(loss_weight * glo_loss_list)

            pred_src = self.cls_List_spec[mark](data_src)

            cls_loss = F.cross_entropy(F.log_softmax(pred_src, dim=1), label_src)

            # mmd
            if mmd_type == 'mmd':
                local_loss += LLMMD.MMD(data_src, data_tgt_spec[mark])
            else:
                # LLMMD
                label_tgt_son = self.cls_List_spec[mark](data_tgt_spec[mark])
                local_loss += LLMMD.LLMMD(data_src, data_tgt_spec[mark], label_src,
                                     torch.nn.functional.softmax(label_tgt_son, dim=1))

            return cls_loss, local_loss, glo_loss

        else:
            data = self.sharedNet(data_src)
            pred = []
            for i in range(self.num_source):
                temp = self.avgpool(self.specificNet[i](data))
                temp = temp.view(temp.size(0), -1)
                pred.append(torch.nn.functional.softmax(self.cls_List_spec[i](temp), dim=1))
            return pred


def eegnet(pretrained=False, **kwargs):
    """Constructs a eegnet model.
    """
    eegnet_model = ComEEGNet()

    return eegnet_model
