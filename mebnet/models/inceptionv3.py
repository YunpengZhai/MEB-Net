from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


__all__ = ['InceptionV3', 'inceptionv3']


class InceptionV3(nn.Module):

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(InceptionV3, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        inception_v3 = torchvision.models.inception_v3(pretrained=True)

        self.base = nn.Sequential(
            inception_v3.Conv2d_1a_3x3, inception_v3.Conv2d_2a_3x3, inception_v3.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2), 
            inception_v3.Conv2d_3b_1x1, inception_v3.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
            inception_v3.Mixed_5b, inception_v3.Mixed_5c, inception_v3.Mixed_5d, 
            inception_v3.Mixed_6a, inception_v3.Mixed_6b, inception_v3.Mixed_6c, inception_v3.Mixed_6d, inception_v3.Mixed_6e, 
            inception_v3.Mixed_7a, inception_v3.Mixed_7b, inception_v3.Mixed_7c)
        self.gap = nn.AdaptiveAvgPool2d(1)


        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = 2048

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        # if not pretrained:
        #     self.reset_params()

    def forward(self, x, feature_withbn=False):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x, prob
        return x, prob



def inceptionv3(**kwargs):
    return InceptionV3(50, **kwargs)

