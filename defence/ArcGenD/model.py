import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class ArcGenD(nn.Module):
    def __init__(self, input_size, class_num, num_query=20):
        super(ArcGenD, self).__init__()
        self.input_size = input_size
        self.class_num = class_num
        self.num_query = num_query

        self.inp = nn.Parameter(torch.zeros(self.num_query, *input_size).normal_()*1e-3)
        
        self.feture_extractor = FeatureExtractor(n_feature=self.num_query*self.class_num)
        self.detector = Detector(n_feature=self.num_query*self.class_num, n_class=1)

    def forward(self, pred, norm_flag):
        feature = self.feture_extractor(pred, norm_flag)
        score = self.detector(feature, norm_flag)
        return score, feature

class FeatureExtractor(nn.Module):
    def __init__(self, n_feature):
        super(FeatureExtractor, self).__init__()
        
        self.alignment_layer_fc = nn.Linear(n_feature, n_feature*4)
        self.alignment_layer_fc.weight.data.normal_(0, 0.005)
        self.alignment_layer_fc.bias.data.fill_(0.1)
        self.alignment_layer = nn.Sequential(
            self.alignment_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.alignment_layer_fc2 = nn.Linear(n_feature*4, n_feature)
        self.alignment_layer_fc2.weight.data.normal_(0, 0.005)
        self.alignment_layer_fc2.bias.data.fill_(0.1)
        self.alignment_layer2 = nn.Sequential(
            self.alignment_layer_fc2,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = input.view(input.size(0), -1)
        feature = self.alignment_layer(feature)
        feature = self.alignment_layer2(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Detector(nn.Module):
    def __init__(self, n_feature, n_class=1, n_hidden=20):
        super(Detector, self).__init__()
        self.n_hidden = n_hidden
        self.classifier_layer = nn.Linear(n_feature, self.n_hidden)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.output =  nn.Linear(self.n_hidden, n_class)
        self.output.weight.data.normal_(0, 0.01)
        self.output.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            self.output.weight.data = l2_norm(self.output.weight, axis=0)
            emb = self.classifier_layer(input)
            classifier_out = self.output(emb)
        else:
            emb = self.classifier_layer(input)
            classifier_out = self.output(emb)
        return classifier_out    

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply

class RevGrad(torch.nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)


class Discriminator(nn.Module):
    def __init__(self, n_feature, n_domain, alpha=1.):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(n_feature, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, n_domain)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )

        self.grl_layer = RevGrad()

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1)
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out
    
    