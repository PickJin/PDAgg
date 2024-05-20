import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import pandas as pd

from model.models import FewShotModel

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.8):
        super().__init__()
        init_values = 1e-4
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)

    def forward(self, q, k, v):

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.bmm(q, k.transpose(1, 2))
        #attn = attn / self.temperature
        attn = attn / 25

        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class ScaledDotProductAttention2(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.8):
        super().__init__()
        init_values = 1e-4
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)

    def forward(self, q, k, v):

        #q = F.normalize(q, dim=-1)
        #k = F.normalize(k, dim=-1)

        #attn = torch.bmm(q, k.transpose(1, 2))
        #attn = attn / self.temperature
        #attn = attn / 13

        q = q.unsqueeze(2)
        k = k.unsqueeze(1)

        attn = -((q-k)**2).sum(dim=3)
        attn = attn / 13

        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.8):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        init_values = 1e-4

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention2(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.gamma_1 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)     #best epoch 149, best test acc=0.6976, best epoch 111,  best test auc=0.7204
        self.gamma_2 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head         #尝试下pre_norm
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        #q = self.norm(q)
        #k = self.norm(k)
        #v = self.norm(v)      #pre_norm 1
        """ q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)            #去掉这三个线性映射: best epoch 89, best test acc=0.7006, best epoch 55,  best test auc=0.7160
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) """
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output*self.gamma_1 + residual)

        return output

class MultiHeadAttention2(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.8):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        init_values = 1e-4

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        #self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.attention = ScaledDotProductAttention(12)
        self.layer_norm = nn.LayerNorm(d_model)

        self.gamma_1 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        """ q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) """

        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        #output = self.dropout(self.fc(output))
        #output = self.layer_norm(output*self.gamma_1 + residual)       #残差连接很重要，单要这一个和一个attention，test acc=0.6988 (>0.69), best epoch 41,  best test auc=0.7161 (>0.7)，且数值都比较稳定
        output = output*self.gamma_1 + residual

        return output

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))
    
class FEAT(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
            hdim2 = 16000
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        else:
            raise ValueError('')
        
        self.args = args
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)   
        self.slf_attn2 = MultiHeadAttention2(1, hdim, hdim, hdim, dropout=0.5) 
        #self.slf_attn3 = MultiHeadAttention3(1, hdim, hdim, hdim, dropout=0.5)  
        #self.slf_attn2 = ScaledDotProductAttention(13)    
        init_values = 0.04

        self.channel_weight = nn.Parameter(init_values * torch.ones((1, 1, 640)),requires_grad=True)

        #self.classifier = CC_head(hdim, 5)
        self.conv2 = nn.Conv2d(5, 25, 1, stride=1, padding=0)
        self.conv1 = ConvBlock(25, 5, 1)
        #self.fc = nn.Linear(640, 1)
        #self.fc.apply(init_weights)
        #self.alpha = nn.Parameter(torch.tensor([0.25, 0.75]).unsqueeze(1).unsqueeze(1).unsqueeze(1), requires_grad=True)

        #self.conv2_s = nn.Conv2d(5, 25, 1, stride=1, padding=0)
        #self.conv1_s = ConvBlock(25, 5, 1)

    def get_masks(self, a):
        a = a.unsqueeze(0) #torch.Size([1, 5, 150, 25, 25])
        #a = (a.max(3))[0]  #torch.Size([1, 5, 150, 25])
        a = a.mean(3)
        """ a = a.transpose(1, 3) 
        a = F.relu(self.conv1(a))
        a = self.conv2(a) 
        a = a.transpose(1, 3) """
        a = torch.sigmoid(a/0.025)
        return a

    def get_masks_support(self, a):
        a = a.unsqueeze(0) #torch.Size([1, 5, 150, 25, 25])
        #a = (a.max(4))[0]  #torch.Size([1, 5, 150, 25])
        a = a.mean(4)
        """ a = a.transpose(1, 3) 
        a = F.relu(self.conv1(a))
        a = self.conv2(a) 
        a = a.transpose(1, 3) """
        a = torch.sigmoid(a/0.025)
        return a
        
    def _forward(self, instance_embs, dense_feature, support_idx, query_idx, open_query_idx, aug=False, self_comp=False, index=None):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view( *(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view( *(query_idx.shape   + (-1,)))
        open_query   = instance_embs[open_query_idx.contiguous().view(-1)].contiguous().view( *(open_query_idx.shape   + (-1,)))

        if not self.training:
            #index = np.random.randint(0,20)
            s = pd.DataFrame(support.reshape(-1, 640).detach().cpu().numpy())
            kq = pd.DataFrame(query.reshape(-1, 640).detach().cpu().numpy())
            oq = pd.DataFrame(open_query.reshape(-1, 640).detach().cpu().numpy())
            s.to_csv('/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save_emb/'+str(index)+'_s.csv', index=False)
            kq.to_csv('/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save_emb/'+str(index)+'_kq.csv', index=False)
            oq.to_csv('/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save_emb/'+str(index)+'_uq.csv', index=False)

        ################################################
        if self.args.use_dense:
            support_dense = dense_feature[support_idx.contiguous().view(-1)].contiguous().view( *(support_idx.shape + (640, 5, 5))).squeeze(0)   #support torch.Size([1, 5, 640, 5, 5])
            #support = self.slf_attn2(support, support, support)
            #support = support.mean(dim=1).view( *(support_idx.shape + (-1,)))
            #support = support.view(support.shape[0],-1).view( *(support_idx.shape + (-1,)))
            query_dense = dense_feature[query_idx.contiguous().view(-1)].contiguous().view( *(query_idx.shape   + (640, 5, 5))).squeeze(0)     #query torch.Size([15, 5, 640, 5, 5])
            open_query_dense = dense_feature[open_query_idx.contiguous().view(-1)].contiguous().view( *(open_query_idx.shape   + (640, 5, 5))).squeeze(0)

            #query = query.mean(dim=3)
            #open_query = open_query.mean(dim=3)
            query_all = torch.cat([query_dense, open_query_dense], dim=0)    #all torch.Size([30, 5, 640, 5, 5])

            support_dense = support_dense.view(self.args.shot, self.args.way, -1 ).view(self.args.way*self.args.shot, -1)   #torch.Size([5, 16000])
            query_all = query_all.view(self.args.way* self.args.query*2, -1 )

            support_embedding = support_dense.view(self.args.way*self.args.shot, 640, -1)     #sup_emb torch.Size([5, 640, 25])
            query_embedding = query_all.view(self.args.way* self.args.query*2, 640, -1)

            support_embedding_norm = F.normalize(support_embedding, p=2, dim=1, eps=1e-12)
            query_embedding_norm = F.normalize(query_embedding, p=2, dim=1, eps=1e-12)

            multi_level = False
            if multi_level:
                kernal_size = [2,4]
                support_embedding_map = support_embedding.view(self.args.way*self.args.shot, 640, 5, 5)
                query_embedding_map = query_embedding.view(self.args.way* self.args.query*2, 640, 5, 5)
                #池化
                pool_list = []
                for size in kernal_size:
                    support_embedding_pooled = F.avg_pool2d(support_embedding_map, size, stride=1)
                    query_embedding_pooled = F.avg_pool2d(query_embedding_map, size, stride=1)

                    support_embedding_pooled = support_embedding_pooled.view(self.args.way*self.args.shot, 640, -1)
                    query_embedding_pooled = query_embedding_pooled.view(self.args.way* self.args.query*2, 640, -1)

                    support_embedding_pnorm = F.normalize(support_embedding_pooled, p=2, dim=1, eps=1e-12)
                    query_embedding_pnorm = F.normalize(query_embedding_pooled, p=2, dim=1, eps=1e-12)

                    support_embedding_pnorm = support_embedding_pnorm.transpose(1, 2).unsqueeze(1)
                    query_embedding_pnorm = query_embedding_pnorm.unsqueeze(0)

                    semantic_correlations_pooled = torch.matmul(support_embedding_pnorm, query_embedding_pnorm)

                    pool_query_guided_masks = self.get_masks(semantic_correlations_pooled)
                    refined_pool_support_embedding = support_embedding_pooled.unsqueeze(0).unsqueeze(2)* pool_query_guided_masks.unsqueeze(3)
                    refined_pool_support_embedding = refined_pool_support_embedding.squeeze(0).transpose(0,1).reshape(self.args.way* self.args.query*2, self.args.shot,self.args.way, -1).mean(1)
                    refined_pool_support_embedding = refined_pool_support_embedding.view(self.args.way* self.args.query*2, self.args.way, 640, (6-size)**2).mean(3)
                    pool_list.append(refined_pool_support_embedding.unsqueeze(0))

            support_embedding_norm = support_embedding_norm.transpose(1, 2).unsqueeze(1)   #torch.Size([5, 1, 25, 640])
            query_embedding_norm = query_embedding_norm.unsqueeze(0)        #torch.Size([1, 150, 640, 25])
            support_embedding_norm2 = support_embedding_norm.transpose(2, 3).squeeze(1).unsqueeze(0)

            semantic_correlations = torch.matmul(support_embedding_norm, query_embedding_norm)  #torch.Size([5, 150, 25, 25])
            semantic_correlations_support = torch.matmul(support_embedding_norm, support_embedding_norm2)  #torch.Size([5, 150, 25, 25])

            query_guided_masks = self.get_masks(semantic_correlations)  #torch.Size([1, 5, 150, 25])
            support_guided_masks = self.get_masks_support(semantic_correlations)

            #query_guided_masks = F.softmax(query_guided_masks*0.025, dim=3)*25
            #support_guided_masks = F.softmax(support_guided_masks*0.025, dim=3)*25

            support_masks = self.get_masks(semantic_correlations_support)
            support_masks2 = self.get_masks_support(semantic_correlations_support)

            refined_support_embedding = support_embedding.unsqueeze(0).unsqueeze(2)* query_guided_masks.unsqueeze(3)  #torch.Size([1, 5, 150, 640, 25])
            refined_support_embedding = support_embedding.unsqueeze(0).unsqueeze(2) + refined_support_embedding

            refined_support_embedding = refined_support_embedding.squeeze(0).transpose(0,1).reshape(self.args.way* self.args.query*2, self.args.shot,self.args.way, -1).mean(1)
            #refined_support_embedding = F.normalize(refined_support_embedding, p=2, dim=-1, eps=1e-12)

            refined_query_embedding = query_embedding.unsqueeze(0).unsqueeze(1)* support_guided_masks.unsqueeze(3)
            refined_query_embedding = refined_query_embedding.squeeze(0).transpose(0,1).reshape(self.args.way* self.args.query*2, self.args.shot,self.args.way, -1).mean(1)

            support_support_embedding = support_embedding.unsqueeze(0).unsqueeze(2)* support_masks.unsqueeze(3)
            support_support_embedding = support_support_embedding.squeeze(0).transpose(0,1).reshape(self.args.way* self.args.shot, self.args.shot,self.args.way, -1).mean(1)

            support_support_embedding2 = support_embedding.unsqueeze(0).unsqueeze(1)* support_masks2.unsqueeze(3)
            support_support_embedding2 = support_support_embedding2.squeeze(0).transpose(0,1).reshape(self.args.way* self.args.shot, self.args.shot,self.args.way, -1).mean(1)
            #query = query.view(query.shape[0], query.shape[1], query.shape[2], -1)
            #open_query = open_query.view(open_query.shape[0], open_query.shape[1], open_query.shape[2], -1)
            ################################################

            refined_support_embedding = refined_support_embedding.view(self.args.way* self.args.query*2, self.args.way, 640, 25).mean(3)
            refined_query_embedding = refined_query_embedding.view(self.args.way* self.args.query*2, self.args.way, 640, 25).mean(3)

            support_support_embedding = support_support_embedding.view(self.args.way* self.args.shot, self.args.way, 640, 25).mean(3)
            support_support_embedding2 = support_support_embedding2.view(self.args.way* self.args.shot, self.args.way, 640, 25).mean(3)
            if multi_level:
                alpha = torch.tensor([0.23, 0.08, 0.69]).unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
                pool_list.append(refined_support_embedding.unsqueeze(0))
                pool_list = torch.cat(pool_list, dim=0)                
                refined_support_embedding = (pool_list*alpha).sum(0)

        # get mean of the support
        if self.args.use_dense:
            proto = support.mean(dim=1) # Ntask x NK x d
        else:
            if aug:
                proto = support[0].unsqueeze(0).mean(dim=1) ##################
                #proto = support.mean(dim=1)
            else:
                proto = support.mean(dim=1)

        """ proto = F.normalize(proto, dim=-1)
        query = F.normalize(query, dim=-1).view(proto.shape[0], -1, emb_dim)
        open_query = F.normalize(open_query, dim=-1).view(proto.shape[0], -1, emb_dim)

        logits = torch.bmm(query, proto.permute([0,2,1])) 
        logits = logits.view(-1, proto.shape[1])

        logits_open = torch.bmm(open_query, proto.permute([0,2,1])) 
        logits_open = logits_open.view(-1, proto.shape[1])

        return logits, logits_open  """

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        """ beta = 0.5

        proto = F.relu(proto)
        open_query = F.relu(open_query)
        query = F.relu(query) 

        proto = torch.pow(proto, beta)
        open_query = torch.pow(open_query, beta)
        query = torch.pow(query, beta)  """
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        b, q, w, e = query.shape
        query = query.view(b, -1, e)
        b1, q1, w1, e1 = open_query.shape
        open_query = open_query.view(b1, -1, e1)
        #proto = self.slf_attn(proto, proto, proto)
        refined_support_embedding = refined_support_embedding.view(-1, 640).unsqueeze(0)
        refined_query_embedding = refined_query_embedding.view(-1, 640).unsqueeze(0)
        support_support_embedding = support_support_embedding.view(-1, 640).unsqueeze(0)
        if self.training:
            #query = self.slf_attn(query, proto, proto)
            #open_query = self.slf_attn(open_query, proto, proto) 

            #refined_support_embedding = self.slf_attn(refined_support_embedding, proto, proto)     #torch.Size([1, 5, 640])

            """ a = refined_support_embedding
            a2 = support_support_embedding
            a3 = ((torch.abs(a)).mean()/(torch.abs(a2)).mean())
            print('d', a3)
            refined_support_embedding = self.slf_attn(refined_support_embedding, support_support_embedding, support_support_embedding)
            d = refined_support_embedding
            print('kkl', a-d)
            c = (torch.abs(a)/torch.abs(d)).mean()
            print('c', c) """ 
            print('b', support_support_embedding.shape)
            refined_support_embedding = self.slf_attn(refined_support_embedding, support_support_embedding, support_support_embedding)
            #refined_query_embedding = self.slf_attn(refined_query_embedding, support_support_embedding, support_support_embedding)
            #refined_support_embedding = self.slf_attn(refined_support_embedding, refined_support_embedding, refined_support_embedding)
            support_support_embedding = self.slf_attn(support_support_embedding, support_support_embedding, support_support_embedding)
            """ query_all = torch.cat([query, open_query], dim=1)
            print('ss', query_all.shape)
            print('kk', proto.shape)
            query_all
            query_all = self.slf_attn3(proto, query_all, query_all)
            proto = self.slf_attn3(query_all, proto, proto)
            query = query_all[:, :75]
            open_query = query_all[:, 75:] """

            if aug:
                support = support.view(-1, support.shape[2], support.shape[3])
                support = self.slf_attn(support, support, support)
            #proto = self.slf_attn2(proto, proto, proto) 
            #proto2 = self.slf_attn(proto, query, query)   
            #query = self.slf_attn(query, proto, proto)
            #open_query = self.slf_attn(open_query, proto, proto)
        else:
            """ query_all = torch.cat([query, open_query], dim=1)
            #open_query = self.slf_attn(open_query, proto, proto)
            #query = self.slf_attn(query, proto, proto)
            query_all = self.slf_attn(query_all, proto, proto)
            query = query_all[:, :75]
            open_query = query_all[:, 75:] """
            refined_support_embedding = self.slf_attn(refined_support_embedding, support_support_embedding, support_support_embedding)
            #support_support_embedding = self.slf_attn(support_support_embedding, support_support_embedding, support_support_embedding)
            #refined_query_embedding = self.slf_attn(refined_query_embedding, support_support_embedding, support_support_embedding)
            #refined_query_embedding = self.slf_attn(refined_query_embedding, proto, proto)

            """ query_all = torch.cat([query, open_query], dim=1)
            query_all = self.slf_attn3(proto, query_all, query_all)
            proto = self.slf_attn3(query_all, proto, proto)
            query = query_all[:, :75]
            open_query = query_all[:, 75:] """

            if aug: 
                support = support.view(-1, support[2], support[3])
                support = self.slf_attn(support, support, support)
            #query = self.slf_attn2(query, proto, proto)
            #proto = self.slf_attn2(proto, proto, proto) 
            #query = self.slf_attn(query, proto, proto)
            #open_query = self.slf_attn(open_query, proto, proto)
        '''
        query = query.view(b, q, w, e)
        proto = self.slf_attn(proto, query, query)   
        query = self.slf_attn(query, proto, proto)  
        '''
        query = query.view(b, q, w, e)
        open_query = open_query.view(b1, q1, w1, e1)

        refined_support_embedding = refined_support_embedding.view(self.args.way* self.args.query*2, self.args.way, 640)
        refined_query_embedding = refined_query_embedding.view(self.args.way* self.args.query*2, self.args.way, 640)

        clip = torch.quantile(support_support_embedding, 0.98)  #tiered 1
        refined_query_embedding = torch.clip(refined_query_embedding, min=None, max=clip)

        close_support = refined_support_embedding[:self.args.way* self.args.query]
        open_support = refined_support_embedding[self.args.way* self.args.query:]

        refined_query_embedding = refined_query_embedding.view(self.args.way* self.args.query*2, self.args.way, 640)
        refined_close_query = refined_query_embedding[:self.args.way* self.args.query]
        refined_open_query = refined_query_embedding[self.args.way* self.args.query:]

        support_support_embedding = support_support_embedding.view(self.args.way* self.args.shot, self.args.way, 640)
        support_support_embedding2 = support_support_embedding2.view(self.args.way* self.args.shot, self.args.way, 640)
        #if self.training and self.args.use_euclidean:
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            open_query = open_query.view(-1, emb_dim).unsqueeze(1)

            #proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            #proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
            #pair_represent = (refined_support_embedding-refined_query_embedding)**2

            #logits_output = self.fc(pair_represent)
            
            # logits = torch.exp(- torch.sum((close_support - refined_close_query) ** 2, 2) / 64)
            # logits_open = torch.exp( - torch.sum((open_support - refined_open_query) ** 2, 2) / 64)

            match_pair = pd.DataFrame()
            unmatch_pair = pd.DataFrame()
            unmatch_pair = []
            known = (close_support - refined_close_query) 
            unknown = (open_support - refined_open_query) 


            known = known.view(-1,640).detach().cpu().numpy()
            unknown = unknown.view(-1, 640).detach().cpu().numpy()
    
            known = pd.DataFrame(known)
            unknown = pd.DataFrame(unknown)
            """ for m in range(15):
                for n in range(5):
                    for q in range(5):
                        if n==q:
                            match_pair[str(m*5+n)] = known[m*5+n][q]
                        else:
                            unmatch_pair.append(pd.DataFrame(columns=[str(m*5*5+n*5+q)], data=[known[m*5+n][q]]))
                            #unmatch_pair[str(m*5*5+n*5+q)] = known[m*5+n][q]

            un_all = pd.DataFrame(unknown)
            unmatch_pair.append(un_all)
            unmatch_pair = pd.concat(unmatch_pair, axis=1)

            match_pair.to_csv('/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save/match.csv')
            unmatch_pair.to_csv('/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save/unmatch_pair.csv') """
            
            if not self.training:
                known.to_csv('/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save/'+str(index)+'_match.csv', index=False)
                unknown.to_csv('/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save/'+str(index)+'_unmatch.csv', index=False)

            f_open = close_support - refined_close_query
            f_close = close_support - refined_close_query
            print('a', f_open.shape)
            logits = - torch.sum((close_support - refined_close_query) ** 2, 2) / 64
            logits_open = - torch.sum((open_support - refined_open_query) ** 2, 2) / 64

            #logits = torch.sum(torch.exp(-((close_support - refined_close_query) ** 2) / 3), 2)
            #logits_open =  torch.sum(torch.exp(-((open_support - refined_open_query) ** 2) / 3), 2)

            #logits = -logits_output[:self.args.way* self.args.query].squeeze()
            #logits_open = -logits_output[self.args.way* self.args.query:].squeeze()

            if self_comp:
                #proto_dump = proto.clone().squeeze().unsqueeze(1)
                logits_comp = - torch.sum((support_support_embedding - support_support_embedding2) ** 2, 2) / self.args.temperature

            if aug:
                support = support.view(-1, emb_dim).unsqueeze(1)
                logits_aug = - torch.sum((proto - support) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
            open_query = open_query.view(num_batch, -1, emb_dim)
            #query_n = F.normalize(query, dim=-1) # (Nbatch,  Nq*Nw, d)
            #open_query_n = F.normalize(open_query, dim=-1)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

            logits_open = torch.bmm(open_query, proto.permute([0,2,1])) / self.args.temperature
            logits_open = logits_open.view(-1, num_proto)

            if self_comp:
                #proto_dump = proto.clone().squeeze().unsqueeze(1)
                logits_comp = - torch.sum((support_support_embedding - support_support_embedding2) ** 2, 2) / self.args.temperature

            if aug:
                support = support.view(-1, emb_dim).unsqueeze(1)
                logits_aug = - torch.sum((proto - support) ** 2, 2) / self.args.temperature
        
        # for regularization
        if self.training:
            if aug:
                return logits, logits_open, None, logits_aug, (close_support, open_support, support_support_embedding, refined_close_query, refined_open_query)

            if self_comp:
                return logits, logits_open, None, logits_comp, (close_support, open_support, support_support_embedding, refined_close_query, refined_open_query)

            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
                # aux_task = F.normalize(aux_task, dim=-1)       ########
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            if aug:
                return logits, logits_open, logits_reg, logits_aug
            else:
                return logits, logits_open, logits_reg            
        else:
            return logits, logits_open, (close_support, open_support, support_support_embedding, refined_close_query, refined_open_query)


def weight_norm(module, name='weight', dim=0):

    WeightNorm.apply(module, name, dim)
    return module

def init_weights(m):
    nn.init.constant(m.weight, val=1)
    nn.init.constant(m.bias, val=0)

class CC_head(nn.Module):
    def __init__(self, indim, outdim,scale_cls=10.0, learn_scale=True, normalize=True):
        super().__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=learn_scale
        )
        self.normalize=normalize

    def forward(self, features):
        if features.dim() == 4:
            if self.normalize:
                features=F.normalize(features, p=2, dim=1, eps=1e-12)
            features = F.adaptive_avg_pool2d(features, 1).squeeze_(-1).squeeze_(-1)
        assert features.dim() == 2
        x_normalized = F.normalize(features, p=2, dim=1, eps = 1e-12)
        self.L.weight.data = F.normalize(self.L.weight.data, p=2, dim=1, eps = 1e-12)
        cos_dist = self.L(x_normalized)
        classification_scores = self.scale_cls * cos_dist

        return classification_scores

class MultiHeadAttention3(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.8):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        init_values = 1e-4

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention3(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.gamma_1 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)     #best epoch 149, best test acc=0.6976, best epoch 111,  best test auc=0.7204
        self.gamma_2 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head         #尝试下 pre_norm
        num_c, len_q, _ = q.size()
        num_c, len_k, _ = k.size()
        num_c, len_v, _ = v.size()

        residual = q
        #q = self.norm(q)
        #k = self.norm(k)
        #v = self.norm(v)      #pre_norm 1
        """ q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)            #去掉这三个线性映射: best epoch 89, best test acc=0.7006, best epoch 55,  best test auc=0.7160
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.view(num_c, len_q, n_head, d_k)
        k = k.view(num_c, len_k, n_head, d_k)
        v = v.view(num_c, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv """

        output, attn, log_attn = self.attention(q, k, v)

        print('output', output.shape)
        output = output.sum(-1, keepdim=True)
        output = torch.bmm(output, v) + k

        """ output = output.view(n_head, num_c, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(num_c, len_q, -1) # b x lq x (n*dv) """

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output*self.gamma_1 + residual)

        return output

class ScaledDotProductAttention3(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.8):
        super().__init__()
        init_values = 1e-4
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((640)),requires_grad=True)

    def forward(self, q, k, v):

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.bmm(q, k.transpose(1, 2))
        #attn = attn / self.temperature
        attn = attn / 25

        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn
