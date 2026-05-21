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

from core.data import get_data_info
from core.data import load_data
from core.data import DATASETS
from core.models import create_model
from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core.utils import str2bool, str2float
from core.attacks import HNLinfPGDAttack, LinfPGDAttack

from SelfPure import SelfPure
from AutoattackforSelfPure import AutoAttack
# from AutoattckforSelfPure

from robustbench.utils import load_model



parser = argparse.ArgumentParser(description='the eval about No auxiliary model adversarial purification for robustness model.')

parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training.')

parser.add_argument('--data-dir', type=str, default='./data/')
parser.add_argument('--classnum', type=int, default=10)
parser.add_argument('--prefix', type=str, default='compare')
parser.add_argument('--log-dir', type=str, default='./log/')
parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.')
    
parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
# parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')



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

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'

DATA_DIR = args.data_dir + da


log_path = LOG_DIR + '/eval_AA_AdvSm.log'
# logger = Logger(log_path)
# print_args(args=args)
# print_args(args=args,logger=logger)

seed(args.seed)

#data

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.data in ['cifar10', 'cifar10s']:
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
elif args.data in ['cifar100', 'cifar100s']:
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)  

# info = get_data_info(DATA_DIR)
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size_validation
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size
# _, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False,shuffle_train=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logger.log('Using device: {}'.format(device))

model = load_model(model_name=args.desc, dataset=args.data, threat_model='Linf').to(device)
model.eval()


seed(args.seed)

l = [x for (x, y) in test_dataloader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_dataloader]
y_test = torch.cat(l, 0)

adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard',log_path=log_path,seed=0)
X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=50)