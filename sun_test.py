from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
import csv
import model as model
import util
import classifier as classifier
from config import opt
 

if opt.seed is None:
    opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.seed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

opt.num_classes = opt.nclass_all

# save_file = opt.acc_save_file
# with open(save_file, 'a+') as file_write_obj:
#      print_word = 'Random Seed:' + str(opt.seed)
#     file_write_obj.writelines(print_word)
#     file_write_obj.write('\n')
# print("Save file ", opt.acc_save_file)

netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1_label(opt)
netDec = model.AttDec_atten(opt,opt.attSize, data.attribute.cuda())



print(netE)
print(netG)
print(netD)
print(netDec)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize) #attSize class-embedding size
input_lab = torch.FloatTensor(opt.batch_size)


noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
default_device = torch.cuda.current_device()
print("default_device: {}".format(default_device))
print("opt.cuda: {}".format(opt.cuda))
if opt.cuda:
    netD.cuda()
    netE.cuda()
    # netF.cuda()
    netG.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_lab = input_lab.cuda()
    one = one.cuda()
    mone = mone.cuda()

resume_netDec = "./save_models/netDec_0.4474.model"
resume_netG = "./save_models/netG_0.4474.model" 
print("test on SUN")

best_gzsl_acc = 0

def model_load(resume_path, target_model, best_gzsl_acc):
    if os.path.isfile(resume_path):
        model_dict = target_model.state_dict()
        checkpoint = torch.load(resume_path)
        trained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        if (best_gzsl_acc==0):
            best_gzsl_acc = checkpoint['best_prec1']
        target_model.load_state_dict(trained_dict)
        return target_model, best_gzsl_acc
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))



print("resume_netDec {} ".format(resume_netDec))
netDec, best_gzsl_acc = model_load(resume_netDec, netDec, 0)
print("netDec load successfully! ")
print("resume_netG {} ".format(resume_netG))
netG, best_gzsl_acc = model_load(resume_netG, netG, 0)
print("netG load successfully! ")



def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_att, batch_label = data.next_seen_batch(opt.batch_size)
    test_id, idx = np.unique(batch_label, return_inverse=True)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_lab.copy_(batch_label)

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)

def generate_syn_feature(generator,classes, attribute,num,netF=None,netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        fake = generator(syn_noisev,c=syn_attv)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerDec       = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates, _, _ = netD(interpolates, Variable(input_att), input_label)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss, self).__init__()
        # self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix
    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.uint8)
        mask = mask.cuda()
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        instance_zone = torch.exp((self.remove_diag(similarity_matrix) - margin)/temperature)
        inst2proxy_positive = torch.exp((self.cosine_similarity(inst_embed, proxy) - margin)/temperature)
        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels.type(torch.long)]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)
            inst2inst_positives = instance_zone*mask_4_remove_negatives
            numerator = (inst2inst_positives.sum(dim=1)+inst2proxy_positive)
        else:
            numerator = inst2proxy_positive
        denomerator = torch.cat([torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(numerator/denomerator).mean()
        return criterion


class A2C_loss(torch.nn.Module):
    def __init__(self, bias=0.4):
        super(A2C_loss, self).__init__()
        print("A2C_loss bias ", bias)
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.bias = bias

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.uint8)
        mask = mask.cuda()
        return M[mask].view(h, -1)
    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    def forward(self, inst_embed, labels, inst_proxy, labels_proxy, margin, alpha, real_list, is_real, att_distance):
        """
        Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        """
        dist_mat_ = self.calculate_similarity_matrix(inst_embed, inst_proxy)
        dist_mat = 1 - dist_mat_
        N = dist_mat.size(0) 
        total_loss = list()
        for ind in range(N):
            distance_semantic = att_distance[labels[ind]][:] # 200 
            alpha_dis = distance_semantic[labels_proxy] # labels_proxy length 
            is_pos = labels_proxy.eq(labels[ind])
            is_neg = labels_proxy.ne(labels[ind])
            if is_real:
                is_pos = is_pos * real_list
                is_neg = is_neg * real_list
            dist_ap = dist_mat[ind][is_pos]
            dist_an = dist_mat[ind][is_neg]
            dist_att_neg = alpha_dis[is_neg]
            alpha = dist_att_neg * 0.5 + self.bias # 0.4
            alpha = alpha.cuda()
            ap_is_pos = torch.clamp(torch.add(dist_ap, -0.05),min=0.0)
            ap_pos_num = ap_is_pos.size(0) +1e-5
            ap_pos_val_sum = torch.sum(ap_is_pos)
            loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num)) # positive 
            an_is_pos = torch.lt(dist_an,alpha) # shi fou dist_an<alpha
            an_less_alpha = dist_an[an_is_pos]
            alpha_minming = alpha[an_is_pos]
            an_is_neg = alpha_minming - an_less_alpha
            an_dist_lm = torch.sum(an_is_neg)
            ap_neg_num = an_is_neg.size(0) + 1e-5
            loss_an = an_dist_lm
            loss_an = torch.div(an_dist_lm, float(ap_neg_num))
            total_loss.append(loss_ap + loss_an)
        total_loss_all = sum(total_loss) * 1.0/N
        return total_loss_all

import random
class M_Bank:
    def __init__(self, opt):
        self.K = 1000
        self.s_num = opt.lat_num
        feats_length = opt.ngh
        print("the length of memory is {} , and the selected number is {} ".format(self.K, self.s_num))
        self.feats = torch.zeros(self.K, feats_length).cuda()
        self.targets = -torch.ones(self.K, dtype=torch.long).cuda()
        self.real_list = -torch.ones(self.K, dtype=torch.uint8).cuda()
        self.ptr = 0
    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            idx0 = range(0, self.K)
            idx_select = random.sample(idx0, self.s_num)
            return self.feats[idx_select], self.targets[idx_select], self.real_list[idx_select]
        elif self.ptr < self.s_num:
            return self.feats[:self.ptr], self.targets[:self.ptr], self.real_list[:self.ptr]
        else:
            idx0 = range(0, self.ptr)
            idx_select = random.sample(idx0, self.s_num)
            return self.feats[idx_select], self.targets[idx_select], self.real_list[idx_select]

    def enqueue_dequeue(self, feats, targets, real_list):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.real_list[-q_size:] = real_list
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.real_list[self.ptr: self.ptr + q_size] = real_list
            self.ptr += q_size



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



cudnn.benchmark = True # 10-19
best_gzsl_acc = 0
best_zsl_acc = 0
margin = 0

netG.eval()
netDec.eval()

print("opt.syn_num {} opt.classifier_lr {} opt.seed {} ".format(opt.syn_num, opt.classifier_lr, opt.seed))
syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num,netF=None,netDec=netDec)
if opt.gzsl:   
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    nclass = opt.nclass_all
    gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
            60, opt.syn_num, generalized=True, netDec=netDec, dec_size=1024, dec_hidden_size=opt.ngh)
    print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))
