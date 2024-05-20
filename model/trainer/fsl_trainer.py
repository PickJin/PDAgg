import time
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torchvision import transforms

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
    cal_auc,
    cal_auc_temp,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm

from PIL import ImageFilter
import random
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
from sklearn.metrics import pairwise_distances_argmin_min, f1_score

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        # self.evaluate_test()

        label, label_aux = self.prepare_label()
        test_auc = []
        test_acc = []
        test_aucp = []
        test_accp = []
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            tauc = Averager()

            start_tm = time.time()
            if (epoch)%2==0:
                _, va, vap, vauc, vaucp = self.evaluate_test()
                test_auc.append(vauc)
                test_aucp.append(vaucp)
                test_acc.append(va)
                test_accp.append(vap)

            self.model.train()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]
                
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                if self.args.data_aug:
                    data = self.data_aug(data)
                    logits, logits_open, reg_logits, logits_aug, refined_embedding = self.para_model(data, aug=True)
                elif self.args.data_comp:
                    logits, logits_open, reg_logits, logits_comp, refined_embedding = self.para_model(data, self_comp=True)
                else:
                    logits, logits_open, reg_logits, refined_embedding = self.para_model(data, aug=False)

                if reg_logits is None:
                    loss1 = F.cross_entropy(logits, label)
                    label_dup = label.unsqueeze(1)
                    binary_label_close = torch.zeros(logits.shape).cuda()
                    binary_label_close = binary_label_close.scatter_(1, label_dup, 1)
                    #binary_label_close = F.softmax(binary_label_close*2, dim=-1)
                    #binary_label_close = binary_label_close.scatter_(1, label_dup, 1)

                    binary_label_open = torch.zeros(logits_open.shape).cuda()

                    loss9 = F.binary_cross_entropy(torch.sigmoid((logits+11.1)), binary_label_close)
                    loss10 = F.binary_cross_entropy(torch.sigmoid((logits_open+11.1)), binary_label_open)

                    #loss9 = F.binary_cross_entropy(torch.sigmoid((logits)), binary_label_close)
                    #loss10 = F.binary_cross_entropy(torch.sigmoid((logits_open)), binary_label_open)

                    #close_logits_open2 = logits_open.clone()[:, :args.way]
                    #open_pred_log = torch.log(torch.softmax(logits_open, dim=1))
                    #loss7 = torch.sum(open_pred_log)/open_pred_log.shape[-1]/open_pred_log.shape[-2]     ##开集数据在闭集标签上得分尽量平均
                    #loss7 = torch.sum(torch.log(torch.sum(torch.exp(logits_open), dim=1)))

                    #m_in = 8.5
                    #m_out = 9.5
                    m_in = 9
                    m_out = 10
                    Ec_out = -torch.logsumexp(logits_open, dim=1)
                    Ec_in = -torch.logsumexp(logits, dim=1)
                    #print('out', Ec_out)
                    #print('in', Ec_in)

                    #m_in2 = 7.6 mini-1shot
                    #m_out2 = 9.4 mini-1shot
                    m_in2 = 9
                    m_out2 = 11
                    Ec_out2 = -torch.log(torch.sum(torch.log(1+torch.exp(logits_open)), dim=-1) + 1e-6)
                    Ec_in2 = -torch.log(torch.sum(torch.log(1+torch.exp(logits)), dim=-1) + 1e-6)
                    #print('out', Ec_out2)
                    #print('in', Ec_in2)
                    loss7 = 1*(torch.pow(F.relu(Ec_in2-m_in2), 2).mean() + torch.pow(F.relu(m_out2-Ec_out2), 2).mean())

                    #loss7 = 1*(torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean())
                    #loss7 = 0.1*(torch.pow(Ec_in, 2).mean() + torch.pow(-Ec_out, 2).mean())

                    dummy_label1 = torch.ones(logits.shape[0])
                    dummy_label2 = torch.zeros(logits_open.shape[0])
                    dummy_label = torch.cat([dummy_label1, dummy_label2], dim=0).cuda()

                    logits = logits.view(self.args.query, self.args.way, self.args.way)
                    logits = logits.permute(1, 0, 2).contiguous()
                    logits = logits.view(-1, self.args.way)

                    pred_close = []
                    for i in range(self.args.way):
                        pred_close.append(logits[i*self.args.query:(i+1)*self.args.query, i])

                    pred_close = torch.cat(pred_close)

                    pred_close = logits.max(dim=1)[0]
                    pred_open = logits_open.max(dim=1)[0]
                    pred = torch.cat([pred_close, pred_open], dim=0)
                    pred = F.softmax(pred,dim=0)

                    #logits_aug = torch.tensor([i for i in range(125)]).view(25,5).cuda()

                    loss2 = F.cross_entropy(pred.unsqueeze(0).float(), dummy_label.unsqueeze(0).float())

                    pred_close = pred_close.unsqueeze(0).repeat(logits_open.shape[0],1).view(-1, 1)
                    pred_open = pred_open.unsqueeze(1).repeat(1, logits.shape[0]).view(-1, 1)
                    pred_conj = torch.cat([pred_close, pred_open], dim=1)

                    dummy_label2 = torch.tensor([1, 0]).unsqueeze(0).repeat(logits_open.shape[0]*logits.shape[0], 1).cuda()

                    loss3 = F.cross_entropy(pred_conj.float(), dummy_label2.float())    

                    #query to query loss
                    pred_q_ins = []
                    pred_q_ref = []
                    for i in range(self.args.way):
                        pred_q_ins.append(logits[i*self.args.query:(i+1)*self.args.query, i].unsqueeze(0))
                        q_ref = torch.cat([logits[:i*self.args.query, i],logits[(i+1)*self.args.query:, i]]).unsqueeze(0)
                        pred_q_ref.append(q_ref)

                    pred_q_ins = torch.cat(pred_q_ins)
                    pred_q_ref = torch.cat(pred_q_ref)

                    pred_q_ins = pred_q_ins.unsqueeze(1).repeat(1, (self.args.way-1)*self.args.query, 1).view(-1, 1)
                    pred_q_ref = pred_q_ref.unsqueeze(2).repeat(1, 1, self.args.query).view(-1, 1)
                    pred_q_exemplar = torch.cat([pred_q_ins, pred_q_ref], dim=1)

                    dummy_label6 = torch.tensor([1, 0]).unsqueeze(0).repeat((self.args.way-1)*self.args.query*self.args.query*self.args.way, 1).cuda()

                    loss6 = F.cross_entropy(pred_q_exemplar.float(), dummy_label6.float())
                                  
                    #loss3 = torch.sum(-pred_close)
                    if self.args.data_aug:
                        pred_aug_ins = []
                        pred_aug_ref = []
                        for i in range(self.args.way):
                            pred_aug_ins.append(logits_aug[i*(self.args.way*self.args.shot):(i+1)*(self.args.way*self.args.shot), i].unsqueeze(0))
                            aug_ref = torch.cat([logits_aug[:i*(self.args.way*self.args.shot), i],logits_aug[(i+1)*(self.args.way*self.args.shot):, i]]).unsqueeze(0)
                            pred_aug_ref.append(aug_ref)

                        pred_aug_ins = torch.cat(pred_aug_ins)
                        pred_aug_ref = torch.cat(pred_aug_ref)

                        pred_aug_ins = pred_aug_ins.unsqueeze(1).repeat(1, (self.args.aug_num-1)*self.args.way, 1).view(-1, 1)
                        pred_aug_ref = pred_aug_ref.unsqueeze(2).repeat(1, 1, self.args.way).view(-1, 1)
                        pred_exemplar = torch.cat([pred_aug_ins, pred_aug_ref], dim=1)

                        dummy_label3 = torch.tensor([1, 0]).unsqueeze(0).repeat((self.args.aug_num-1)*self.args.aug_num*(self.args.way**2)*(self.args.shot**2), 1).cuda()

                        loss4 = F.cross_entropy(pred_exemplar.float(), dummy_label3.float()) 

                    if self.args.data_comp:
                        pred_ins = []
                        pred_ref = []
                        for i in range(self.args.way):
                            pred_ins.append(logits[i*(self.args.query):(i+1)*(self.args.query), i].unsqueeze(0))
                            aug_ref = torch.cat([logits_comp[:i, i],logits_comp[(i+1):, i]]).unsqueeze(0)
                            pred_ref.append(aug_ref)
                        
                        pred_ins = torch.cat(pred_ins, dim=0)
                        pred_ref = torch.cat(pred_ref, dim=0)

                        pred_ins = pred_ins.unsqueeze(1).repeat(1, (self.args.way-1), 1).view(-1, 1)
                        pred_ref = pred_ref.unsqueeze(2).repeat(1, 1, self.args.query).view(-1, 1)
                        #pred_exemplar = torch.cat([pred_ins, pred_ref], dim=1)

                        #dummy_label5 = torch.tensor([1, 0]).unsqueeze(0).repeat((self.args.way-1)*self.args.query*self.args.way, 1).cuda()

                        #loss5 = F.cross_entropy(pred_exemplar.float(), dummy_label5.float())
                    
                    #根据文中的公式再写一个loss

                    #total_loss = loss1 + loss2 + args.balance * F.cross_entropy(reg_logits, label_aux)
                    #total_loss = loss1 + 2*loss2
                    #total_loss = loss1 + 2*loss5 + loss2 + loss3
                    #total_loss = loss1 + 3*loss2 + loss3 + 0.5*loss7   #2,2, best epoch 40, best test acc=0.6964 + 0.0082, best epoch 186,  best test auc=0.7309 + 0.0082
                    total_loss =  0*loss2 + 1*loss3 + 0.2*loss7 + loss9 + loss10
                    #total_loss =  3*loss2 + loss3 + 2*loss7 + loss9 + loss10
                    #total_loss = loss1
                    #total_loss = loss1 + 3*loss2 + loss3 + 0.5*loss7
                    #total_loss =  loss9 + loss10
                    #total_loss =  3*loss2 + loss3 + 0.5*loss7 + loss1
                    #total_loss =  loss1 + 3*loss2 + loss3 + 0.5*loss7
                    #total_loss = loss1 + 3*loss2 + loss3

                else:
                    loss1 = F.cross_entropy(logits, label)

                    close_logits_open2 = logits_open.clone()[:, :args.way]
                    open_pred_log = torch.log(torch.softmax(close_logits_open2, dim=1))
                    loss4 = -torch.sum(open_pred_log)/open_pred_log.shape[-1]/open_pred_log.shape[-2]     ##开集数据在闭集标签上得分尽量平均
                    
                    total_loss = loss4 + F.cross_entropy(logits, label)
                

                tl2.add(loss1)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)

                tl1.add(total_loss.item())

                #close_acc
                acc = count_acc(logits, label)
                ta.add(acc)

                #openset_auc
                auc = cal_auc(logits, logits_open)
                tauc.add(auc)


                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            #if epoch%5 == 0:
            #    self.evaluate_test()

            print('Epoch {}  ETA:{}/{}'.format(
                    epoch,
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        result = pd.DataFrame()
        result['test_auc'] = test_auc
        result['test_acc'] = test_acc
        result['test_aucp'] = test_aucp
        result['test_accp'] = test_accp

        result.to_csv(osp.join(self.args.save_path, '{}-{}-{}-{}.csv'.format(self.args.way, self.args.lr, self.args.lr_mul, self.args.use_euclidean)))

        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 3)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits, logits_open, refined_embedding = self.model(data, self_comp=True)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                auc = cal_auc(logits, logits_open)

                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                record[i-1, 2] = auc
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        vauc, vaucp = compute_confidence_interval(record[:,2])
        
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap, vauc, vaucp

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        #self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model = self.para_model
        self.model.eval()
        record = np.zeros((args.num_test_episodes, 4)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits, logits_open, refined_embedding = self.model(data, self_comp=True)

                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                #print('logits',logits.shape, F.softmax(logits,dim=1))
                #print('logits_open', logits.shape, F.softmax(logits_open,dim=1))
                (close_support, open_support, support_support_embedding, refined_close_query, refined_open_query) = refined_embedding

                
                # energy + recat
                embedding_numpy = torch.cat([refined_close_query, refined_open_query], dim=0).detach().cpu().numpy()
                close_support = close_support.detach().cpu().numpy()
                open_support = open_support.detach().cpu().numpy()
                support_support_embedding = support_support_embedding.detach().cpu().numpy()

                clip = np.quantile(support_support_embedding, 0.97)
                embedding_all = np.clip(embedding_numpy, a_min=None, a_max=clip)

                refined_close_query = embedding_all[:self.args.way* self.args.query]
                refined_open_query = embedding_all[self.args.way* self.args.query:]

                #logits = - np.sum((close_support - refined_close_query) ** 2, 2) / self.args.temperature
                #logits_open = - np.sum((open_support - refined_open_query) ** 2, 2) / self.args.temperature

                auc_logits2 = torch.max(logits, dim=1)[0].cpu().detach().numpy()
                auc_logits_open2 = torch.max(logits_open, dim=1)[0].cpu().detach().numpy()

                logits = logits.detach().cpu().numpy()
                logits_open = logits_open.detach().cpu().numpy()
                auc_logits = logsumexp(logits, axis=-1)
                auc_logits_open = logsumexp(logits_open, axis=-1)
                auc_logits = -np.sum(np.log(1-1/(1+np.exp(-logits))), axis=-1)
                auc_logits_open = -np.sum(np.log(1-1/(1+np.exp(-logits_open))), axis=-1)
                auc = cal_auc_temp(auc_logits, auc_logits_open)
                a = pd.DataFrame()
                
                a['a'] = auc_logits2
                a['b'] = auc_logits_open2
                #a.to_csv('/disk/8T/jinj/FSL/FSOR/tmp10/FEAT-master/save2/'+str(i)+'.csv')

                #f1_score
                #threshold = -10
                threshold = 4.5e-05   #最优
                # threshold = 7e-05
                label_f1 = label.clone().detach().cpu().numpy()
                all_labels = np.concatenate([label_f1, self.args.way * np.ones(self.args.way*self.args.query)], -1).astype(np.int)
                auc_logits_all = np.concatenate([auc_logits, auc_logits_open], axis=0)
                logits_all = np.concatenate([logits, logits_open], axis=0)
                ypred = np.argmax(logits_all, axis=-1) 

                open_jug = (auc_logits_all>threshold)
                #print('a', auc_logits)
                #print('b', auc_logits_open)
                ypred = (ypred*open_jug)+(1-open_jug)*self.args.way
                #print('a', auc_logits_all)
                #print('k', all_labels)
                #print('m', ypred)

                f1score = f1_score(all_labels, ypred, average='macro', labels=np.unique(ypred))
        
                #vim
                '''logits_all = torch.cat([logits, logits_open], dim=0)
                pred = torch.argmax(logits_all, dim=1)
                embedding_pick = []
                for k in range(len(pred)):
                    embedding_pick.append(embedding_numpy[k][pred[k]])
                embedding_numpy = np.array(embedding_pick)

                ec = EmpiricalCovariance(assume_centered=True)
                support_support_embedding = support_support_embedding.reshape(-1, support_support_embedding.shape[-1])
                ec.fit(support_support_embedding)    ####
                eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
                NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[512:]]).T)

                vlogit_id_train = norm(np.matmul(support_support_embedding, NS), axis=-1)        #####
                logits = logits.detach().cpu().numpy()
                logits_open = logits_open.detach().cpu().numpy()
                logits_all = np.concatenate([logits, logits_open])
                alpha = logits_all.max(axis=-1).mean() / vlogit_id_train.mean()
                #a, b = embedding_numpy.shape[0], embedding_numpy.shape[1]
                #embedding_numpy = embedding_numpy.reshape(-1, embedding_numpy.shape[-1])
                vlogit_id_val = norm(np.matmul(embedding_numpy, NS), axis=-1) * alpha   ######

                energy_id_val = logsumexp(logits_all, axis=-1)
                score_id = -vlogit_id_val + energy_id_val
                #score_id = score_id.reshape(a, b).min(-1)

                auc = cal_auc_temp(score_id[:self.args.way* self.args.query], score_id[self.args.way* self.args.query:])'''

                """ #KL-matching
                #comp_logits
                logits_all = torch.cat([logits, logits_open])
                prob_all = F.softmax(logits_all, dim=-1)
                logits_all = logits_all.detach().cpu().numpy()
                prob_all = prob_all.detach().cpu().numpy()

                pred_labels = np.argmax(prob_all, axis=-1)
                structure_score = []
                logits_comp = logits_comp.detach().cpu().numpy()
                logits_comp_copy = []
                for k in range(logits_comp.shape[0]):
                    logits_comp_copy.append( self.softmax(np.concatenate([logits_comp[k][:k], logits_comp[k][k+1:]]) ))
                
                logits_comp = np.array(logits_comp_copy)

                for k in range(logits_all.shape[0]):
                    max_logits = logits_all[k][pred_labels[k]]
                    aux_logits = np.concatenate([logits_all[k][:pred_labels[k]], logits_all[k][pred_labels[k]+1:]])
                    aux_logits_now = np.array([self.softmax(aux_logits)])
                    logits_comp_now = np.array([logits_comp[pred_labels[k]]])
                    #print(k)
                    #print(aux_logits_now)
                    #print(logits_comp_now)
                    aux_score = -pairwise_distances_argmin_min(aux_logits_now, logits_comp_now, metric=self.kl)[1] 
                    #print('a', max_logits)
                    #print('b', aux_score)
                    #print(logits_comp)
                    score = aux_score[0]
                    structure_score.append(score)

                structure_score = np.array(structure_score)
                auc = cal_auc_temp(structure_score[:self.args.way* self.args.query], structure_score[self.args.way* self.args.query:]) """

                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                record[i-1, 2] = auc
                record[i-1, 3] = f1score

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])

        vauc, vaucp = compute_confidence_interval(record[:,2])
        vf1, vf1p = compute_confidence_interval(record[:,3])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl
        self.trlog['test_auc'] = vauc
        self.trlog['test_auc_interval'] = vaucp
        self.trlog['test_f1'] = vf1
        self.trlog['test_f1_interval'] = vf1p

        if va >= self.trlog['test_max_acc']:
            self.trlog['test_max_acc'] = va
            self.trlog['test_max_acc_interval'] = vap
            self.trlog['test_max_acc_epoch'] = self.train_epoch
            self.save_model('test_max_acc')
            
        if vauc >= self.trlog['test_max_auc']:
            self.trlog['test_max_auc'] = vauc
            self.trlog['test_max_auc_interval'] = vaucp
            self.trlog['test_max_auc_epoch'] = self.train_epoch
            self.save_model('test_max_auc')

        if vf1 >= self.trlog['test_max_f1']:
            self.trlog['test_max_f1'] = vf1
            self.trlog['test_max_f1_interval'] = vf1p
            self.trlog['test_max_f1_epoch'] = self.train_epoch
            self.save_model('test_max_f1')


        print('best epoch {}, best test acc={:.4f} + {:.4f}, best epoch {},  best test auc={:.4f} + {:.4f}, best epoch {},  best test f1={:.4f} + {:.4f}\n'.format(
                self.trlog['test_max_acc_epoch'],
                self.trlog['test_max_acc'],
                self.trlog['test_max_acc_interval'],
                self.trlog['test_max_auc_epoch'],
                self.trlog['test_max_auc'],
                self.trlog['test_max_auc_interval'],
                self.trlog['test_max_f1_epoch'],
                self.trlog['test_max_f1'],
                self.trlog['test_max_f1_interval'],
                ))

        print('Test acc={:.4f} + {:.4f},  Test auc={:.4f} + {:.4f}, Test f1={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval'],
                self.trlog['test_auc'],
                self.trlog['test_auc_interval'],
                self.trlog['test_f1'],
                self.trlog['test_f1_interval'],))

        return vl, va, vap, vauc, vaucp
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))     


    def task_adaptive(self, model, data):
        args = self.args
        top_para = [v for k,v in model.named_parameters() if 'encoder' not in k]     ###

        optimizer = optim.SGD(
            [{'params': top_para, 'lr': args.adaptive_lr}],
            lr=args.adaptive_lr,
            momentum=args.adaptive_mom,
            nesterov=True,
            weight_decay=args.adaptive_weight_decay
        )

        #这里需要进行数据增强
        data = self.data_aug(data)

        with TemporaryGrad():
            loss = model(data, adaptive=True)
            #loss = F.cross_entropy(logits, label)  #计算loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model
    
    def data_aug(self, data):
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.Resize(84),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply(transforms.GaussianBlur(3, sigma=(0.1, 2.0)), p=0.5),    ########
            transforms.RandomHorizontalFlip(),
        ])
        support = data[:self.args.eval_way*self.args.eval_shot]
        query = data[self.args.eval_way*self.args.eval_shot:]

        aug_data = [support]
        for i in range(self.args.aug_num-1):
            aug = augmentation(support)
            aug_data.append(aug)
    
        aug_data.append(query)
        data = torch.cat(aug_data, dim=0)
        #print('query', data.shape)

        return data

    def kl(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def softmax(self, fo):
        f = fo.copy()
        # instead: first shift the values of f so that the highest number is 0:
        f -= np.max(f) # f becomes [-666, -333, 0]
        return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer



class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
