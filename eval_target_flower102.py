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
from core.metrics import accuracy

from SelfPure import SelfPure

# from robustbench.utils import load_model
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from timm.models import xcit
from timm.models import create_model



parser = argparse.ArgumentParser(description='the eval about No auxiliary model adversarial purification for robustness model.')

parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training.')

parser.add_argument('--data-dir', type=str, default='/data/rzha795/data/')
parser.add_argument('--classnum', type=int, default=102)
parser.add_argument('--prefix', type=str, default='target')
parser.add_argument('--log-dir', type=str, default='/data/rzha795/adv-smoothing/log/')

parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
# parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')

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
parser.add_argument('--agg-type',type=str,choices=['mean','sycn'],default="mean",help="adv-smooth-type(or agg type)")
parser.add_argument('--keep-original-x',action='store_true',default=False,help='if keep original x during smoothing')
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

if 'resnet' in args.desc:
    model = create_model('resnet50',num_classes = 102,in_chans=3)
    checkpoint = torch.load('./models/flowers102/Linf/resnet_50-Oxford Flowers.pth.tar')
    model.load_state_dict(checkpoint)
elif 'xcit' in args.desc:
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
                           eps_iter=args.sattack_step, clip_min=0.0, clip_max=1.0, targeted=args.sattack_target, 
                           rand_init=args.sattack_rand,HN_regular=args.sattack_HN,noise_eps=args.sattack_HNeps)

def target_eval(model,dataloader,attack,device,eval_attack=None):
    clean_acc = 0
    cr_acc=0
    ar_acc=0
    adv_acc=0
    metrics = pd.DataFrame()
    #获取数据
    for item in dataloader:
        x , y = item
        x, y = x.to(device), y.to(device)
    
        x_adv, _ = eval_attack.perturb(x, y)

        with torch.no_grad():
        #repeat数据进行所有目标攻击
            y_adv = model(x_adv)
            y_clean = model(x)

        clean_acc += accuracy(y,y_clean)
        print('cle x acc is {}'.format(clean_acc))


        adv_acc += accuracy(y,y_adv)
        print('raw x acc is {}'.format(adv_acc))
        # print("adv_x score")

        # y_index = torch.softmax(y_adv,dim=1).argmax(dim=1)
        x_tmp = [x[i].repeat(102,1,1,1) for i in range(len(x))]
        x_repeat = torch.cat(x_tmp,dim=0)

        x_adv_tmp = [x_adv[i].repeat(102,1,1,1) for i in range(len(x_adv))]
        x_adv_repeat = torch.cat(x_adv_tmp,dim=0)

        y_tmp = [j for i in range(len(y)) for j in range(102)]
        y_repeat = torch.LongTensor(y_tmp).to(device)


        x_repeat_attack_out, x_delta_repeat = attack.perturb(x_repeat,y_repeat)
        x_adv_repeat_attack_out, x_adv_delta_repeat = attack.perturb(x_adv_repeat,y_repeat)

        x_delta_norm = torch.norm(x_delta_repeat.reshape([len(x_delta_repeat),-1]),p=2,dim=1)
        x_adv_delta_norm = torch.norm(x_adv_delta_repeat.reshape([len(x_adv_delta_repeat),-1]),p=2,dim=1)


        #test target attack result
        with torch.no_grad():
            x_repeat_out = model(x_repeat_attack_out)
            x_adv_repeat_out = model(x_adv_repeat_attack_out)
        
        y_clean_pred = torch.max(y_clean,dim=1)[1]
        y_clean_prob = torch.max(y_clean.softmax(dim=1),dim=1)[0]

        y_adv_pred = torch.max(y_adv,dim=1)[1]
        y_adv_prob = torch.max(y_adv.softmax(dim=1),dim=1)[0]

        y_repeat_clean_pred = torch.max(x_repeat_out,dim=1)[1]
        y_repeat_clean_prob = torch.max(x_repeat_out.softmax(dim=1),dim=1)[0]

        y_repeat_adv_pred = torch.max(x_adv_repeat_out,dim=1)[1]
        y_repeat_adv_prob = torch.max(x_adv_repeat_out.softmax(dim=1),dim=1)[0] 
        
        batch_metrics = {'ture_label': torch.cat([i.repeat(102) for i in y]).cpu().numpy(),
                         'y_clean_label': torch.cat([i.repeat(102) for i in y_clean_pred]).cpu().numpy(),
                         'y_clean_prob': torch.cat([i.repeat(102) for i in y_clean_prob]).cpu().numpy(),
                         'y_adv_label': torch.cat([i.repeat(102) for i in y_adv_pred]).cpu().numpy(),
                         'y_adv_prob': torch.cat([i.repeat(102) for i in y_adv_prob]).cpu().numpy(),
                         'repeat_true_label': y_repeat.cpu().numpy(), 
                         'repeat_clean_label': y_repeat_clean_pred.cpu().numpy(),
                         'repeat_clean_prob': y_repeat_clean_prob.cpu().numpy(),
                         'repeat_adv_label': y_repeat_adv_pred.cpu().numpy(),
                         'repeat_adv_prob': y_repeat_adv_prob.cpu().numpy(),
                         'x_delta_norm': x_delta_norm.cpu().numpy(),
                         'x_adv_delta_norm': x_adv_delta_norm.cpu().numpy()
                         }
        # metrics = metrics.append(pd.DataFrame(batch_metrics),ignore_index=True)
        metrics = pd.concat([metrics, pd.DataFrame(batch_metrics)],ignore_index=True,sort=False)


        cr_acc += accuracy(y_repeat,x_repeat_out)
        ar_acc += accuracy(y_repeat,x_adv_repeat_out)
        print('clean repeat x acc is {}'.format(cr_acc))
        print('adv repeat x acc is {}'.format(ar_acc))
    print(clean_acc/len(dataloader))
    print(adv_acc/len(dataloader))
    print(cr_acc/len(dataloader))
    return metrics

seed(args.seed)

metrics = target_eval(model,test_dataloader,sattack,device,gattack)

metrics.to_csv(os.path.join(LOG_DIR,csv_name),index=False)