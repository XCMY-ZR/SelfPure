import torch
import numpy as np
import pandas as pd
from core.utils import ctx_noparamgrad_and_eval

import csv

def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum()
    return accuracy.item()


def clamp(input, min=None, max=None):
    """
    Clamp a tensor by its minimun and maximun values.
    """
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input

def restore_x_by_y(data,x,y):
    data[y]=x
    return data

class SelfPure():
    def __init__(self,agg_type='mean',attackforsmooth=None,attack=None,device=None,class_num=10,noise_eps=8./255):
        '''
        No auxiliary model adversarial purification for robustness model
        also called adv smoothing
        Arguments:
            agg_type: the method find the center point of the sample [mean or sycn] (str)
        '''
        super(SelfPure,self).__init__()

        self.agg_type = agg_type
        if self.agg_type == "mean":
            print("Use mean agg type")
            self.agg_fun = self.mean_center
        else:
            raise ValueError('{} agg type is not supported'.format(agg_type))
        self.noise_eps=noise_eps
        print("Use normal noise({})".format(self.noise_eps))
        self.attackforsmooth = attackforsmooth
        if self.attackforsmooth:
            print("Use adv smoothing")
            self.add_noise = attackforsmooth
        else:
            print("Use normal noise({})".format(self.noise_eps))
            def normal_noise(x):
                x = x + self.noise_eps * torch.rand_like(x)
                return clamp(x,0,1)
            self.add_noise = normal_noise
            
        self.attack = attack
        self.device = device
        self.class_num = class_num

    # def normal_noise(self,x,y):

    def mean_center(self,data):
        return data.mean(dim=0)

    def inject_noise(self, data):
        data = data + self.noise_eps * torch.randn_like(data)
        return clamp(data,0,1)
    
    def agg(self,x,y):
        """
        Argument:
            x : test sample [batch size, ...]
            y : the result of model prediction on x [batch size, class num]
        """
        # y_index = torch.softmax(y,dim=1).argmax(dim=1)

        #The number of samples repeated to categories
        x_tmp = [x[i].repeat(self.class_num,1,1,1) for i in range(len(x))]
        x_repeat = torch.cat(x_tmp,dim=0)

        # make targeted label [0,1,,,n]
        y_tmp = [j for i in range(len(y)) for j in range(self.class_num)]
        # y_repeat = torch.cat(y_tmp,dim=0)
        y_repeat = torch.LongTensor(y_tmp).to(self.device)

        if self.attackforsmooth:
            with torch.enable_grad():
                #targeted attack with HN
                x_adv_repeat, _ = self.add_noise.perturb(x_repeat,y_repeat)
            x_re = torch.stack([self.agg_fun(x_adv_repeat[i*self.class_num:self.class_num*(i+1)]) for i in range(len(x))])
        else:
            x_re = self.add_noise(x)
        return x_re

    def evel(self,model,dataloader):
        """
        return pd dataframe which statistic each sample's prediction
        """
        # model.eval()
        raw_clean_acc, pro_clean_acc = 0, 0
        raw_adv_acc, pro_adv_acc = 0, 0
        x_num = 0

        metrics = pd.DataFrame()

        for x , y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with ctx_noparamgrad_and_eval(model):
                # x_adv, _ = self.attack.perturb(x,y)
                x_adv, _ = self.attack.perturb(x)
            with torch.no_grad():
                raw_clean_out = model(x)
                raw_adv_out = model(x_adv)
            
            raw_clean_pred = torch.max(raw_clean_out,dim=1)[1]
            raw_adv_pred = torch.max(raw_adv_out,dim=1)[1]

            raw_clean_acc += accuracy(y, raw_clean_out)
            raw_adv_acc += accuracy(y, raw_adv_out)
            print("-"*30)
            print("raw_clean :{}; raw_adv:{}".format(raw_clean_acc,raw_adv_acc))

            pro_clean_x = self.agg(x=x,y=raw_clean_out)
            pro_adv_x = self.agg(x=x_adv,y=raw_adv_out)

            with torch.no_grad():
                pro_clean_out = model(pro_clean_x)
                pro_adv_out = model(pro_adv_x)
            
            pro_clean_pred = torch.max(pro_clean_out,dim=1)[1]

            pro_adv_pred = torch.max(pro_adv_out,dim=1)[1]

            pro_clean_acc += accuracy(y, pro_clean_out)
            pro_adv_acc += accuracy(y, pro_adv_out)
            
            print("pro_clean :{}; pro_adv:{}".format(pro_clean_acc,pro_adv_acc))
            print("-"*30)

            x_num += len(x)

            batch_metrics = {'true_label': y.cpu().numpy(), 'raw_clean_pred_label': raw_clean_pred.cpu().numpy(), 
                             'raw_adv_pred_label': raw_adv_pred.cpu().numpy(), 
                             'pro_clean_pred_label': pro_clean_pred.cpu().numpy(),
                             'pro_adv_pred_label': pro_adv_pred.cpu().numpy()}
            
            # metrics = metrics.append(pd.DataFrame(batch_metrics),ignore_index=True)
            metrics = pd.concat([metrics,pd.DataFrame(batch_metrics)], ignore_index=True)
        print("#"*32)
        print("raw_clean_acc :{}; raw_adv_acc:{}".format(raw_clean_acc / x_num,raw_adv_acc / x_num))
        print("pro_clean_acc :{}; pro_adv_acc:{}".format(pro_clean_acc / x_num,pro_adv_acc / x_num))
        print("#"*32)
        
        return metrics
    
    def compare_evel(self,model,dataloader,logger=None,csvfile=None):
        """
        return pd dataframe which statistic each sample's prediction
        """
        model.eval()
        raw_clean_acc, pro_clean_acc, rand_clean_acc = 0, 0, 0
        raw_adv_acc, pro_adv_acc, rand_adv_acc = 0, 0, 0
        x_num = 0

        metrics = pd.DataFrame()

        for x , y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with ctx_noparamgrad_and_eval(model):
                # x_adv, _ = self.attack.perturb(x,y)
                x_adv, _ = self.attack.perturb(x)
            with torch.no_grad():
                raw_clean_out = model(x)
                raw_adv_out = model(x_adv)
            
            raw_clean_pred = torch.max(raw_clean_out,dim=1)[1]
            raw_adv_pred = torch.max(raw_adv_out,dim=1)[1]

            raw_clean_acc += accuracy(y, raw_clean_out)
            raw_adv_acc += accuracy(y, raw_adv_out)
            if logger:
                logger.log("-"*30)
                logger.log("raw_clean :{}; raw_adv:{}".format(raw_clean_acc,raw_adv_acc))
            print("-"*30)
            print("raw_clean :{}; raw_adv:{}".format(raw_clean_acc,raw_adv_acc))

            rand_clean_x = self.inject_noise(x)
            rand_adv_x = self.inject_noise(x_adv)

            pro_clean_x = self.agg(x=x,y=raw_clean_out)
            pro_adv_x = self.agg(x=x_adv,y=raw_adv_out)

            with torch.no_grad():
                pro_clean_out = model(pro_clean_x)
                pro_adv_out = model(pro_adv_x)

                rand_clean_out = model(rand_clean_x)
                rand_adv_out = model(rand_adv_x)
            #adv smoothing
            pro_clean_pred = torch.max(pro_clean_out,dim=1)[1]

            pro_adv_pred = torch.max(pro_adv_out,dim=1)[1]

            pro_clean_acc += accuracy(y, pro_clean_out)
            pro_adv_acc += accuracy(y, pro_adv_out)
            #rand smoothing
            rand_clean_pred = torch.max(rand_clean_out,dim=1)[1]
            rand_adv_pred = torch.max(rand_adv_out,dim=1)[1]

            rand_clean_acc += accuracy(y, rand_clean_out)
            rand_adv_acc += accuracy(y, rand_adv_out)
            
            print("rand_clean :{}; rand_adv:{}".format(rand_clean_acc,rand_adv_acc))
            print("pro_clean :{}; pro_adv:{}".format(pro_clean_acc,pro_adv_acc))
            print("-"*30)

            if logger:
                logger.log("rand_clean :{}; rand_adv:{}".format(rand_clean_acc,rand_adv_acc))
                logger.log("pro_clean :{}; pro_adv:{}".format(pro_clean_acc,pro_adv_acc))

            x_num += len(x)

            batch_metrics = {'true_label': y.cpu().numpy(), 'raw_clean_pred_label': raw_clean_pred.cpu().numpy(), 
                             'raw_adv_pred_label': raw_adv_pred.cpu().numpy(), 
                             'pro_clean_pred_label': pro_clean_pred.cpu().numpy(), 
                             'pro_adv_pred_label': pro_adv_pred.cpu().numpy(),
                             'rand_clean_pred_label': rand_clean_pred.cpu().numpy(), 
                             'rand_adv_pred_label': rand_adv_pred.cpu().numpy()
                             }
            #Real-time recording of results
            if csvfile:
                with open(csvfile, 'a+',newline='') as f:
                    writer=csv.DictWriter(f, fieldnames=batch_metrics.keys())
                    # writer = csv.writer(f)
                    for row in zip(*batch_metrics.values()):
                        writer.writerow(dict(zip(batch_metrics.keys(), row)))
                    # writer.writerow(batch_metrics.values())
            
            # metrics = metrics.append(pd.DataFrame(batch_metrics),ignore_index=True)
            metrics = pd.concat([metrics,pd.DataFrame(batch_metrics)],ignore_index=True)
        print("#"*32)
        print("raw_clean_acc :{}; raw_adv_acc:{}".format(raw_clean_acc / x_num,raw_adv_acc / x_num))
        print("rand_clean_acc :{}; rand_adv_acc:{}".format(rand_clean_acc / x_num,rand_adv_acc / x_num))
        print("pro_clean_acc :{}; pro_adv_acc:{}".format(pro_clean_acc / x_num,pro_adv_acc / x_num))
        print("#"*32)

        
        if logger:
            logger.log("raw_clean_acc :{}; raw_adv_acc:{}".format(raw_clean_acc / x_num,raw_adv_acc / x_num))
            logger.log("rand_clean_acc :{}; rand_adv_acc:{}".format(rand_clean_acc / x_num,rand_adv_acc / x_num))
            logger.log("pro_clean_acc :{}; pro_adv_acc:{}".format(pro_clean_acc / x_num,pro_adv_acc / x_num))

        return metrics
    
        # metrics.to_csv(os.path.join(LOG_DIR, 'out_statistic.csv'), index=False)




