import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet, BasicBlock
            self.encoder = ResNet(BasicBlock, [1,1,1,1])
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet(BasicBlock, [1,1,2,2])
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def split_instances(self, data, aug=False):
        args = self.args
        if self.training:
            if aug:
                return  (torch.Tensor(np.arange(args.way*args.shot*args.aug_num)).long().view(args.aug_num, args.shot, args.way), 
                        torch.Tensor(np.arange(args.way*args.shot*args.aug_num, args.way * (args.shot*args.aug_num + args.query))).long().view(1, args.query, args.way),
                        torch.Tensor(np.arange(args.way * (args.shot*args.aug_num + args.query), args.way * (args.shot*args.aug_num + args.query*2))).long().view(1, args.query, args.way))
            else:
                return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                        torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way),
                        torch.Tensor(np.arange(args.way * (args.shot + args.query), args.way * (args.shot + args.query*2))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way),
                     torch.Tensor(np.arange(args.eval_way * (args.eval_shot + args.eval_query), args.eval_way * (args.eval_shot + args.eval_query + args.query))).long().view(1, args.eval_query, args.eval_way))

    def forward(self, x, aug=False, self_comp=False, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)[0]
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs, dense_feature = self.encoder(x)
            #dense_feature = dense_feature.view(dense_feature.shape[0], dense_feature.shape[1], -1)
            #dense_feature = dense_feature.permute(0,2,1)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx, open_query_idx = self.split_instances(x, aug)
            if self.training:
                if aug or self_comp:
                    logits, logits_open, logits_reg, logits_aug, refined_embedding = self._forward(instance_embs, dense_feature, support_idx, query_idx, open_query_idx, aug, self_comp)
                    return logits, logits_open, logits_reg, logits_aug, refined_embedding
                else:
                    logits, logits_open, logits_reg, refined_embedding = self._forward(instance_embs, dense_feature, support_idx, query_idx, open_query_idx, aug, self_comp)
                    return logits, logits_open, logits_reg, refined_embedding
            else:
                #这里后面可以考虑task adaptive
                index = np.random.randint(0,20)
                torch.save(x, '/disk/8T/jinj/FSL/FSOR/code_pre/FEAT-master-1shot/save_emb/'+str(index)+'_img.pt')
                logits, logits_open, refined_embedding = self._forward(instance_embs, dense_feature, support_idx, query_idx, open_query_idx, aug, self_comp, index)
                return logits, logits_open, refined_embedding

    def _forward(self, x, support_idx, query_idx, open_query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')