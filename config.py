#author: akshitac8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--is_fix', dest='is_fix', action='store_true', default=False,
                    help='is_fix.')
parser.add_argument('--num_classes', type=int, help='number of classes', default=200)
parser.add_argument('--sf_size', type=int, help='arrtibute size', default=0)
parser.add_argument('--sf', help='arrtibute', default=0)
# parser.add_argument('--sigma', dest='sigma', default=0.5, type=float,
#                     help='sigma.')
parser.add_argument('--epochs_ft', type=int, default=60, help='number of epochs to train for fine-tune resnet101')
parser.add_argument('--lr_ft', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay_ft', default=30, type=int,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--save_path', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--path_h5', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--acc_save_file', default='save_acc_default', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# add classifier
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")

# contrastive learning
parser.add_argument('--contrastive_lambda', type=float, default=1.0, help='weight for contrastive leanring')
parser.add_argument('--nonlinear_embed', action='store_true', default=False, help='enable nonlinear embedding')
parser.add_argument('--normalize_embed', action='store_true', default=False, help='enable normalization')
parser.add_argument('--tem', type=float, default=1.0, help='temperatures')
parser.add_argument('--tem2', type=float, default=1.0, help='temperatures2')
parser.add_argument('--rank_lambda', type=float, default=1.0, help='weight for rank contrastive leanring')

parser.add_argument('--cos_lambda', type=float, default=1.0, help='weight for rank contrastive leanring')

parser.add_argument('--atten_weight', type=float, default=1.0, help='weight for dynamic semantic correction')


# parser.add_argument('--pretrained', default='True', help='resnet101 pretrained')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument("--conditional", action='store_true',default=True)
###

parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=1.0, help='recons_weight for decoder')
parser.add_argument('--feedback_loop', type=int, default=2)
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
parser.add_argument('--cls_weight', type=float, default=1.0, help='the last classifier in AttDec')
parser.add_argument('--sem_weight', type=float, default=1.0, help='the semantic projection in AttDec')

parser.add_argument('--add_source', action='store_true', default=False, help='If add the original feature to the final feature')
parser.add_argument('--add_latent', action='store_true', default=False, help='If add the latent feature to the final feature, used in ds2m and SPDNet')
parser.add_argument('--add_align', action='store_true', default=False, help='If add the alignment feature to the final feature')
parser.add_argument('--lat_num', type=int, default=200, help='the number of the selected sample from the memory bank')
parser.add_argument('--lat_bias', type=float, default=0.4, help='The bias of the negative class boundary')
parser.add_argument('--att_norm', action='store_true', default=False, help='Use normalize operation to the refined att')

parser.add_argument('--c1', type=float, default=0.5)
parser.add_argument('--finetune', action='store_true', default=False, help='Is the feature the finetuned feature')

opt = parser.parse_args()
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
opt.latent_size = opt.attSize
