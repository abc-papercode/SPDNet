import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator,self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        latent_size=opt.latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid=nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        return x


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder,self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class Discriminator_D1_label(nn.Module):
    def __init__(self, opt): 
        super(Discriminator_D1_label, self).__init__()
        self.nonlinear_embed = opt.nonlinear_embed
        self.normalize_embed = opt.normalize_embed
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
        hypersphere_dim = 512
        self.linear2 = nn.Linear(opt.ndh, hypersphere_dim)
        self.embedding = nn.Embedding(num_embeddings=opt.num_classes, embedding_dim=hypersphere_dim)
        self.activation = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(hypersphere_dim, hypersphere_dim)
        # self.embedding = nn.Linear(in_features=opt.attSize, out_features=hypersphere_dim)
    def forward(self, x, att, label):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        # cls_proxy = self.embedding(label)
        # cls_embed = self.linear2(self.hidden)
        # if self.nonlinear_embed:
        #     cls_embed = self.linear3(self.activation(cls_embed))
        # if self.normalize_embed:
        #     cls_proxy = F.normalize(cls_proxy, dim=1)
        #     cls_embed = F.normalize(cls_embed, dim=1)
        authen_output = self.fc2(self.hidden)
        return authen_output


class AttDec_atten(nn.Module):
    '''
        2021/10/15
        Using the visual features to construct the mapping matrix M1 of attribute a
    '''
    def __init__(self, opt, attSize, attribute):
        super(AttDec_atten, self).__init__()
        self.attribute = attribute
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.attSize = attSize
        self.lrelu = nn.LeakyReLU(0.2, True)
        hypersphere_dim = 1024
        self.embedding = nn.Embedding(num_embeddings=opt.num_classes, embedding_dim=hypersphere_dim)
        self.matrix_size = attSize * attSize
        self.bias_size = attSize
        self.zsr_sem = nn.Sequential(
            nn.Linear(attSize,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(opt.ngh,1024),
            nn.LeakyReLU(),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(opt.ngh,1024),
            nn.LeakyReLU(),
        )
        self.mapping_matrix = nn.Sequential(
            nn.Linear(opt.ngh, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.matrix_size + self.bias_size),
            nn.LeakyReLU(),
            )
        self.apply(weights_init)
    def forward(self, feat, label=None):
        dis_feas = self.lrelu(self.fc1(feat))
        if label is not None:
            cls_proxy = self.embedding(label)
            cls_proxy = F.normalize(cls_proxy, p=2, dim=1)
        cls_embed = self.proj2(dis_feas)
        cls_embed = F.normalize(cls_embed, p=2, dim=1)
        zsr_visfea_1 = self.proj(dis_feas)
        x_norm = F.normalize(zsr_visfea_1, p=2, dim=1)
        if label is not None:
            attribute_batch = self.attribute[label] # 10 * 312
            matrix_bias = self.mapping_matrix(dis_feas)
            matrix = matrix_bias[:,:self.matrix_size].view(-1, self.attSize, self.attSize)
            bias = matrix_bias[:,self.matrix_size:].view(-1, 1, self.attSize)
            att_pro1_ = torch.bmm(attribute_batch.unsqueeze(1), matrix) + bias
            weights_atn = F.softmax(att_pro1_.squeeze(1), dim=1)
            attribute_batch_new = weights_atn * attribute_batch + attribute_batch
            att_pro2 = self.zsr_sem(attribute_batch_new)
            zsr_classifier =  att_pro2
            w_norm = F.normalize(zsr_classifier, p=2, dim=1)
            return cls_embed, cls_proxy, dis_feas, w_norm, x_norm
        else:
            return cls_embed, dis_feas, x_norm 
    def get_zsr_visfea(self):
        return self.x_norm.detach()