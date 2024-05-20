import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)       #数据集中所有数据的标签
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)        #m_ind中的每一个元素代表每一类所有样本的序号

    def __len__(self):
        return self.n_batch      #任务数

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            print('batch',batch)
            yield batch

class OpenCategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per 
        self.n_per_open = 15
        self.n_open = 5
        self.add = 0

        label = np.array(label)       #数据集中所有数据的标签
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)        #m_ind中的每一个元素代表每一类所有样本的序号

    def __len__(self):
        return self.n_batch      #任务数

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch1 = []
            batch2 = []
            index_list = torch.randperm(len(self.m_ind))
            classes1 = index_list[:self.n_cls]
            classes2 = index_list[self.n_cls:(self.n_cls+self.n_open)]      #openset_cls
            for c in classes1:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch1.append(l[pos])
            for c in classes2:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per_open]
                pos_add = torch.randperm(len(l))[self.n_per_open:self.n_per_open+self.add]
                batch2.append(l[pos])

            batch1 = torch.stack(batch1).t().reshape(-1)    
            batch2 = torch.stack(batch2).t().reshape(-1)    ####
            batch2 = torch.cat([batch2, pos_add])
            batch = torch.cat([batch1, batch2], dim=0)
            yield batch

class RandomSampler():

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch
            
            
# sample for each class
class ClassSampler():

    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]
            
            
# for ResNet Fine-Tune, which output the same index of task examples several times
class InSetSampler():

    def __init__(self, n_batch, n_sbatch, pool): # pool is a tensor
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = self.pool[torch.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch