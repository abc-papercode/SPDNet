import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import pdb
import h5py

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1    

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))


        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att, batch_label




class DATA_LOADER_h5(object):
    def __init__(self, path_h5, opt):
        self.read_matdataset(path_h5, opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, path_h5, opt):
        data = h5py.File(path_h5, 'r')
        feats_seen_train = data['seen_features_train'][...]
        labels_seen_train = data['seen_labels_train'][...]
        feats_seen = data['seen_features'][...]
        labels_seen = data['seen_labels'][...]
        feats_unseen = data['unseen_features'][...]
        labels_unseen = data['unseen_labels'][...]
    
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feats_seen_train)
                # print("feats_seen_train shape: {}".format(feats_seen_train.shape))
                # print("feats_seen_train: ", feats_seen_train[0][0:40])

                # print("_train_feature shape: {}".format(_train_feature.shape))
                # print("_train_feature: ", _train_feature[0][0:40])

                _test_seen_feature = scaler.transform(feats_seen)
                _test_unseen_feature = scaler.transform(feats_unseen)
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(labels_seen_train).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(labels_unseen).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(labels_seen).long()

            else:
                self.train_feature = torch.from_numpy(feats_seen_train).float()
                self.train_label = torch.from_numpy(labels_seen_train).long() 
                self.test_unseen_feature = torch.from_numpy(feats_unseen).float()
                self.test_unseen_label = torch.from_numpy(labels_unseen).long() 
                self.test_seen_feature = torch.from_numpy(feats_seen).float() 
                self.test_seen_label = torch.from_numpy(labels_seen).long()
        else:
            self.train_feature = torch.from_numpy(feats_seen_train).float()
            self.train_label = torch.from_numpy(labels_seen_train).long()
            self.test_unseen_feature = torch.from_numpy(feats_unseen).float()
            self.test_unseen_label = torch.from_numpy(labels_unseen).long() 
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))


        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att, batch_label

def Visual_fea_pro(feats, opt):
    if not opt.validation:
        if opt.preprocessing:
            if opt.standardization:
                print('standardization...') 
                scaler = preprocessing.StandardScaler()
            else:
                scaler = preprocessing.MinMaxScaler()
                
            _train_feature = scaler.fit_transform(feats)
            # _test_seen_feature = scaler.transform(feature[test_seen_loc])
            # _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            train_feature = torch.from_numpy(_train_feature).float()
            mx = train_feature.max()
            train_feature.mul_(1/mx)
        else:
            train_feature = torch.from_numpy(feature[trainval_loc]).float()
            train_label = torch.from_numpy(label[trainval_loc]).long() 
            test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
            test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
            test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
    else:
        train_feature = torch.from_numpy(feature[train_loc]).float()
        train_label = torch.from_numpy(label[train_loc]).long()
        test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
        test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
    print("train_feature shape: {}".format(train_feature.shape))
    return train_feature
    

def preprocess_strategy_flip(dataset):
    evaluate_transforms = None
    print("preprocess_strategy flip>>>>>>>>>>")
    train_transforms_flip = transforms.Compose([
        transforms.Resize(480),
        transforms.RandomHorizontalFlip(p=1),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms_flip
    
    
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
