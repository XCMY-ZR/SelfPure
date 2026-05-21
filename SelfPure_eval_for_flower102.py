import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import datetime
import os
import argparse
import torchvision
from torchvision import datasets, transforms

from core.attacks import create_attack
from core.attacks import CWLoss, DLRloss
from core.attacks import ATTACKS


from core.models import create_model
from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core.utils import str2bool, str2float
from core.attacks import HNLinfPGDAttack, LinfPGDAttack

from SelfPure import SelfPure

# from robustbench.utils import load_model
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from timm.models import xcit
from timm.models import create_model



parser = argparse.ArgumentParser(description='the eval about No auxiliary model adversarial purification for robustness model.')

parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training.')

parser.add_argument('--data-dir', type=str, default='./data/')
parser.add_argument('--classnum', type=int, default=102)
parser.add_argument('--prefix', type=str, default='compare')
parser.add_argument('--log-dir', type=str, default='.//log/')
    
parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
# parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')

# # eval-aa.py
# parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
# parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'], 
#                     help='Version of AA.')

# # eval-adv.py
# parser.add_argument('--source', type=str, default=None, help='Path to source model for black-box evaluation.')
# parser.add_argument('--wb', action='store_true', default=False, help='Perform white-box PGD evaluation.')

# # eval-rb.py
# parser.add_argument('--threat', type=str, default='corruptions', choices=['corruptions', 'Linf', 'L2'],
#                     help='Threat model for RobustBench evaluation.')

# eval-smoothing
# parameter for attack that generate adv-example
parser.add_argument('--gattack',type=str,choices=ATTACKS,default='linf-pgd',help="Type of attack")
parser.add_argument('--gattack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
parser.add_argument('--gattack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
parser.add_argument('--gattack-iter', type=int, default=40, help='Max. number of iterations (if any) for the attack.')
parser.add_argument('--gattack-loss',type=str,choices=['cw','ce','dlr'],default='cw',help='loss use for attacking')
parser.add_argument('--gattack-HN',action='store_true',default=False,help='attack use HN regular')
# parameter for attack that smoothing x_input
parser.add_argument('--sattack',type=str,choices=ATTACKS,default='linf-pgd',help="Type of attack")
parser.add_argument('--sattack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
parser.add_argument('--sattack-HNeps', type=str2float, default=4/255, help='Epsilon for the attack.')
parser.add_argument('--sattack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
parser.add_argument('--sattack-iter', type=int, default=20, help='Max. number of iterations (if any) for the attack.')
parser.add_argument('--sattack-loss',type=str,choices=['cw','ce','dlr'],default='cw',help='loss use for attacking')
parser.add_argument('--sattack-HN',action='store_true',default=False,help='attack use HN regular')
parser.add_argument('--sattack-rand',action='store_true',default=False,help='randomly initial the perturbation ')
parser.add_argument('--sattack-target',action='store_true',default=False,help='target attack')
# parameter for agg smoothing
parser.add_argument('--csvfile',action='store_true',default=False)
# mean_eval2_advmodel_stdsample40_norandinit_HNtarget_cw.log


parser.add_argument('--seed', type=int, default=1, help='Random seed.')

args = parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.log('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

LOG_DIR = args.log_dir + '/' + args.desc

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print('Dir path is not exist, {} has been created'.format(LOG_DIR))

# DATA_DIR = ''

geps = str(args.gattack_eps*1000)[:2].replace('.','_')
seps = str(args.sattack_eps*1000)[:2].replace('.','_')
heps = str(args.sattack_HNeps*1000)[:2].replace('.','_')
ghn= "HN" if args.gattack_HN else ""
shn= "{}HN".format(heps) if args.sattack_HN else ""
sat = "with" if args.sattack_target else "no"
ran = "with" if args.sattack_rand else "no"
kex = "keepX" if args.keep_original_x else ""
suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

log_name =  "/log-{}{}-{}{}{}{}{}-{}randint-{}{}{}{}{}-{}target-{}.log".format(args.agg_type, kex, geps, ghn, args.gattack, args.gattack_loss, args.gattack_iter, ran, 
                                                           seps,shn, args.sattack, args.sattack_loss, args.sattack_iter, sat,suffix )
csv_name =  "csv-{}{}-{}{}{}{}{}-{}randint-{}{}{}{}{}-{}target-{}.csv".format(args.agg_type, kex, geps, ghn, args.gattack, args.gattack_loss, args.gattack_iter, ran, 
                                                           seps, shn, args.sattack, args.sattack_loss, args.sattack_iter, sat, suffix)

if args.csvfile:
    csv_file_name =  "csv-{}{}-{}{}{}{}{}-{}randint-{}{}{}{}{}-{}target.csv".format(args.agg_type, kex, geps, ghn, args.gattack, args.gattack_loss, args.gattack_iter, ran, 
                                                           seps, shn, args.sattack, args.sattack_loss, args.sattack_iter, sat)
    csv_file_name=os.path.join(LOG_DIR, args.prefix+csv_file_name)
    print(csv_file_name)
else:
    csv_file_name=None

log_path = LOG_DIR + log_name
logger = Logger(log_path)
print_args(args=args)
print_args(args=args,logger=logger)

seed(args.seed)

#data

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
testset = torchvision.datasets.Flowers102(root=args.data_dir,split='test',download=True,transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)  

# if args.data in ['cifar10', 'cifar10s']:
#     testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
#     test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
# elif args.data in ['cifar100', 'cifar100s']:
#     testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
#     test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)  

# info = get_data_info(DATA_DIR)
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size_validation
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size
# _, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False,shuffle_train=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))

model = create_model('xcit_small_12_p16_224',num_classes = 102,in_chans=3)
checkpoint = torch.load('./models/flowers102/Linf/xcit-s12-Oxford Flowers.pth.tar')
model.load_state_dict(checkpoint)

# model = load_model(model_name=args.desc, dataset=args.data, threat_model='Linf').to(device)
model = normalize_model(model,mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
model.to(device=device)
model.eval()

#attack
LOSS= {
        'ce': nn.CrossEntropyLoss(reduction="sum"),
        'cw': CWLoss,
        'dlr': DLRloss,
}


if args.gattack == 'BPDA':
    # BPDA
    # from advertorch.attacks import LinfPGDAttack
    from advertorch.defenses import MedianSmoothing2D
    from advertorch.defenses import BitSqueezing
    from advertorch.defenses import JPEGFilter
    from advertorch.bpda import BPDAWrapper
    bits_squeezing = BitSqueezing(bit_depth=5)
    median_filter = MedianSmoothing2D(kernel_size=3)
    jpeg_filter = JPEGFilter(10)

    defense = nn.Sequential(
        jpeg_filter,
        bits_squeezing,
        median_filter,
    )
    # BPDA
    defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
    defended_model = nn.Sequential(defense_withbpda, model)

    gattack = LinfPGDAttack(
        defended_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
        nb_iter=1000, eps_iter=2/255, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
    
elif args.gattack == 'BPDA-linf-apgd':
    from advertorch.bpda import BPDAWrapper
    bits_squeezing = BitSqueezing(bit_depth=5)
    median_filter = MedianSmoothing2D(kernel_size=3)
    jpeg_filter = JPEGFilter(10)

    defense = nn.Sequential(
        jpeg_filter,
        bits_squeezing,
        median_filter,
    )
    # BPDA
    defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
    defended_model = nn.Sequential(defense_withbpda, model)

    gattack = create_attack(defended_model, args.gattack_loss, attack_type='linf-apgd', attack_eps=args.gattack_eps, 
                            attack_iter=args.gattack_iter, attack_step=args.gattack_step)

elif args.gattack == 'linf-apgd':
    gattack = create_attack(model, args.gattack_loss, attack_type=args.gattack, attack_eps=args.gattack_eps, 
                            attack_iter=args.gattack_iter, attack_step=args.gattack_step)
    
else:
    # gattack = create_attack(model, CWLoss, 'linf-pgd', 8/255, 40, 2/255)
    gattack = create_attack(model, LOSS[args.gattack_loss], attack_type=args.gattack, attack_eps=args.gattack_eps, 
                            attack_iter=args.gattack_iter, attack_step=args.gattack_step)

if args.gattack_HN:
    print("Bypass test through HN")
    gattack =  HNLinfPGDAttack(model, LOSS[args.gattack_loss], eps=args.gattack_eps, nb_iter=args.gattack_iter, 
                           eps_iter=args.gattack_step, clip_min=0.0, clip_max=1.0, targeted=False, 
                           rand_init=True,HN_regular=True,noise_eps=4./255)

print('Sattack HNeps is {}'.format(args.sattack_HNeps))
sattack =  HNLinfPGDAttack(model, LOSS[args.sattack_loss], eps=args.sattack_eps, nb_iter=args.sattack_iter, 
                           eps_iter=args.sattack_step, clip_min=0.0, clip_max=1.0, targeted=True, 
                           rand_init=False,HN_regular=True,noise_eps=args.sattack_HNeps)
# seed(args.seed)

# seed(args.seed)
# logger.log("rand smoothing")
# advsm = SelfPure(agg_type=args.agg_type,attack=gattack,device=device,keep_original_x=args.keep_original_x)
# re_data_frame = advsm.evel(model,test_dataloader)
# re_data_frame.to_csv(os.path.join(LOG_DIR, "rand"+csv_name), index=False)

# seed(args.seed)
# logger.log("adv smoothing")
# advsm = SelfPure(agg_type=args.agg_type,attackforsmooth=sattack,attack=gattack,device=device,keep_original_x=args.keep_original_x)
# re_data_frame = advsm.evel(model,test_dataloader)
# re_data_frame.to_csv(os.path.join(LOG_DIR, "adv"+csv_name), index=False)

seed(args.seed)
logger.log("compare between rand smoothing and adv smoothing")
advsm = SelfPure(agg_type=args.agg_type,attackforsmooth=sattack,attack=gattack,device=device,,class_num=args.classnum)
re_data_frame = advsm.compare_evel(model,test_dataloader,logger=logger,csvfile=csv_file_name)
re_data_frame.to_csv(os.path.join(LOG_DIR, args.prefix+csv_name), index=False)