import torch
import torch.nn as nn
from transformers import BertModel
from .components import HandshakingKernel

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrained_model_path)
        self.tag_linear = nn.Linear(self.bert.config.hidden_size, 5)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.loss_func = nn.CrossEntropyLoss()
        self.theta = config.theta

    def forward(self, input, attention_mask, token_type_ids, context_mask=None, turn_mask=None, target_tags=None):
        """
        Args:
            input: （batch,seq_len）
            attention_mask: (batch,seq_len)
            token_type_ids: (batch,seq_len)
            context_mask: (batch,seq_len), used to identify labeled tokens
            target_tags: (batch,seq_len)
            turn_mask: (batch,) turn_mask[i]=0 for turn 1，turn_mask[i]=1 for turn 2
        """
        rep, _ = self.bert(input, attention_mask, token_type_ids)
        rep = self.dropout(rep)
        tag_logits = self.tag_linear(rep)  # (batch,seq_len,num_tag)
        if not target_tags is None:
            #其实也可以不去分的吧……
            #区分turn
            tag_logits_t1 = tag_logits[turn_mask == 0]  # (n1,seq_len,num_tag)
            target_tags_t1 = target_tags[turn_mask == 0]  # (n1,seq_len)
            context_mask_t1 = context_mask[turn_mask == 0]  # (n1,seq_len)

            tag_logits_t2 = tag_logits[turn_mask == 1]  # (n2,seq_len,num_tag)
            target_tags_t2 = target_tags[turn_mask == 1]  # (n2,seq_len)
            context_mask_t2 = context_mask[turn_mask == 1]  # (n2,seq_len)

            #区分正文1和问题0
            tag_logits_t1 = tag_logits_t1[context_mask_t1 == 1]  # (N1,num_tag)
            target_tags_t1 = target_tags_t1[context_mask_t1 == 1]  # (N1)

            tag_logits_t2 = tag_logits_t2[context_mask_t2 == 1]  # (N2,num_tag)
            target_tags_t2 = target_tags_t2[context_mask_t2 == 1]  # (N2)

            loss_t1 = self.loss_func(tag_logits_t1, target_tags_t1) if len(
                target_tags_t1) != 0 else torch.tensor(0).type_as(input)
            loss_t2 = self.loss_func(tag_logits_t2, target_tags_t2) if len(
                target_tags_t2) != 0 else torch.tensor(0).type_as(input)
            loss = self.theta*loss_t1+(1-self.theta)*loss_t2
            return loss, (loss_t1.item(), loss_t2.item())
        else:
            # for prediction
            tag_idxs = torch.argmax(
                tag_logits, dim=-1).squeeze(-1)  # (batch,seq_len)
            return tag_idxs



class MrcTpLinkerModel(nn.Module):
    def __init__(self, config):
        super(MrcTpLinkerModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrained_model_path)

        self.handshaking_kernel = HandshakingKernel(self.bert.config.hidden_size,config.shaking_type,config.inner_enc_type)
        self.tag_linear = nn.Linear(self.bert.config.hidden_size, 5)
        self.relH_linear = nn.Linear(self.bert.config.hidden_size, 3)
        self.relT_linear = nn.Linear(self.bert.config.hidden_size, 3)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.loss_func1 = nn.CrossEntropyLoss()#weight=torch.tensor([0.05]+[(1-0.05)/4]*4)
        self.loss_func2 = nn.CrossEntropyLoss(weight=torch.tensor([config.rel_loss_weight]+[(1-config.rel_loss_weight)/2]*2))
        self.theta = config.theta
        self.relH_theta=config.relH_theta
        self.relT_theta=config.relT_theta

    def forward(self, input, attention_mask, token_type_ids, context_mask, turn_mask, target_tags=None, relH_tags=None,relT_tags=None):
        """
        Args:
            input: （batch,seq_len）
            attention_mask: (batch,seq_len)
            token_type_ids: (batch,seq_len)
            context_mask: (batch,seq_len), used to identify labeled tokens
            target_tags: (batch,seq_len)
            turn_mask: (batch,) turn_mask[i]=0 for turn 1，turn_mask[i]=1 for turn 2
        """
        # print(input.shape)
        rep, _ = self.bert(input, attention_mask, token_type_ids)
        rep = self.dropout(rep)
        # 区分turn
        # print(rep.shape)
        rep_t1 = rep[turn_mask == 0]  # (n1,seq_len,num_tag)
        context_mask_t1 = context_mask[turn_mask == 0]  # (n1,seq_len)
        # print(rep_t1.shape,context_mask_t1.shape)

        rep_t2 = rep[turn_mask == 1]  # (n2,seq_len,num_tag)
        context_mask_t2 = context_mask[turn_mask == 1]  # (n2,seq_len)
        # print(rep_t2.shape, context_mask_t2.shape)
        # 区分正文1和问题0
        #t1:普通序列标注
        tag_logits_t1 = self.tag_linear(rep_t1)
        # print(tag_logits_t1.shape,rep_t1.shape)

        # target_tags_t1 = target_tags_t1[context_mask_t1 == 1]  # (N1)
        #t2:rel
        handshaking_rep_t2,handshaking_context_mask=self.handshaking_kernel(rep_t2,context_mask_t2)
        relH_tag_logits = self.relH_linear(handshaking_rep_t2)
        relT_tag_logits = self.relT_linear(handshaking_rep_t2)
        loss_relH = torch.tensor(0).type_as(input)
        loss_relT = torch.tensor(0).type_as(input)
        loss_t1=torch.tensor(0).type_as(input)
        acc=[0,0,0]
        if(relH_tags is not None):
            relH_tag_logits = relH_tag_logits[handshaking_context_mask==1]
            relH_tags_t2 = relH_tags[turn_mask == 1]  # (n2,seq_len)
            relH_tags_t2=relH_tags_t2[relH_tags_t2 != -1]
            loss_relH=self.loss_func2(relH_tag_logits,  relH_tags_t2) if len(
                relH_tags_t2) != 0 else torch.tensor(0).type_as(input)
            # relH_tag_idx = torch.argmax(relH_tag_logits, dim=-1).squeeze(-1) if relH_tag_logits.shape[0] != 0 else torch.tensor(0.0)
            # relT_tag_idx = torch.argmax(relT_tag_logits, dim=-1).squeeze(-1)
            # print(torch.sum(torch.eq(relH_tag_idx,relH_tags_t2)).type_as(input))
            # acc[1]=torch.where(relH_tag_idx>0)[0].shape[0]

        if (relT_tags is not None):
            # print(torch.nonzero(relT_tag_logits>0))
            relT_tag_logits = relT_tag_logits[handshaking_context_mask == 1]
            relT_tags_t2 = relT_tags[turn_mask == 1]  # (n2,seq_len)
            relT_tags_t2=relT_tags_t2[relT_tags_t2!=-1]
            loss_relT = self.loss_func2(relT_tag_logits, relT_tags_t2) if len(
                relT_tags_t2) != 0 else torch.tensor(0).type_as(input)

            # relT_tag_idx = torch.argmax(relT_tag_logits, dim=-1).squeeze(-1) if relT_tag_logits.shape[0] != 0 else torch.tensor(0.0)

            # print( torch.sum(torch.eq(relT_tag_idx, relT_tags_t2)).type_as(input))
            # print(torch.where(relT_tag_idx>0))
            # acc[2] = torch.where(relT_tag_idx>0)[0].shape[0]

        # tag_logits_t2 = tag_logits_t2[context_mask_t2 == 1]  # (N2,num_tag)
        # tag_logits = self.tag_linear(rep)  # (batch,seq_len,num_tag)
        # print(tag_logits.shape)
        # print(target_tags)
        if target_tags is not None:
            tag_logits_t1 = tag_logits_t1[context_mask_t1 == 1]  # (N1,num_tag,类别数量)
            target_tags_t1 = target_tags[turn_mask == 0]  # (n1,seq_len)
            target_tags_t1 = target_tags_t1[context_mask_t1 == 1]
            # target_tags_t1 = target_tags_t1[]
            # print(tag_logits_t1.shape)

            loss_t1 = self.loss_func1(tag_logits_t1, target_tags_t1) if len(
                target_tags_t1) != 0 else torch.tensor(0).type_as(input)
            # target_tags_idx=torch.argmax(tag_logits_t1, dim=-1).squeeze(-1) if tag_logits_t1.shape[0] != 0 else torch.tensor(0.0)
            # acc[0] = torch.where(target_tags_idx>0)[0].shape[0]
        loss = self.relH_theta*loss_relH+self.relT_theta*loss_relT+self.theta*loss_t1
        if(relH_tags is not None or target_tags is not None):
            return loss,(loss_t1.item(),loss_relH.item(),loss_relT.item()),acc
        else:
            # print(tag_logits_t1.shape,relH_tag_logits.shape,relT_tag_logits.shape)
            # print(tag_logits_t1.shape)
            tag_logits_t1 = torch.softmax(tag_logits_t1, dim=-1)
            relH_tag_logits = torch.softmax(relH_tag_logits, dim=-1)
            relT_tag_logits = torch.softmax(relT_tag_logits, dim=-1)
            tag_idx = torch.max(tag_logits_t1, dim=-1) if tag_logits_t1.shape[0] != 0 else torch.tensor(0.0)
            relH_tag_idx = torch.max(relH_tag_logits, dim=-1) if relH_tag_logits.shape[0] != 0 else torch.tensor(0.0)
            relT_tag_idx = torch.max(relT_tag_logits, dim=-1) if relT_tag_logits.shape[0] != 0 else torch.tensor(0.0)
            # print(tag_idx.shape,relH_tag_idx.shape,relT_tag_idx.shape)
            # print(torch.max(tag_logits_t1,dim=-1)[0].shape)
            # print(tag_idx.shape)
            # print(tag_idx)
            return tag_idx, relH_tag_idx, relT_tag_idx, handshaking_context_mask
        #
        #     loss = self.theta*loss_t1+(1-self.theta)*loss_t2
        #     return loss, (loss_t1.item(), loss_t2.item())
        # else:
        #     # for prediction
        #     tag_idxs = torch.argmax(
        #         tag_logits, dim=-1).squeeze(-1)  # (batch,seq_len)
        #     return tag_idxs

