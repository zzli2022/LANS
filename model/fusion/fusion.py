import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets.preprossing import *
from config import *
from copy import deepcopy
import copy
import math

args = get_parser()
src_lang = SrcLang(args.vocab_src_path)
NUM_PATCHES = (args.diagram_size // args.img_patch_size) * (args.diagram_size // args.img_patch_size)

# 只更新视觉token的GroupAttention实现，和只更新文本的GroupAttention实现
def mm_src_to_mask(src): # 应该是batch_max_len+visual_token
    src = src.cpu().detach().numpy() 
    batch_data_mask_tok = []
    for encode_sen_idx in src:
        token = 1
        image_patch_id = 900
        mask = [0] * (len(encode_sen_idx)+NUM_PATCHES)   # 这个mask形式也要变化, 至少变成一个简单的mask
        for num in range(len(encode_sen_idx)+NUM_PATCHES):
            if num<NUM_PATCHES:
                mask[num] = image_patch_id
                image_patch_id += 1
                continue
            mask[num] = token
            if (encode_sen_idx[num-64] == src_lang.word2index[","] or encode_sen_idx[num-64] == src_lang.word2index["."]) \
                    and num != len(encode_sen_idx) - 1: 
                token += 1
            if encode_sen_idx[num-64]==0:mask[num] = 0
        for num in range(len(encode_sen_idx)+NUM_PATCHES): # question part, no effective 
            if mask[num] == (token-1) and token != 1:
                mask[num] = 1000
        batch_data_mask_tok.append(mask) 
    # import pdb
    # pdb.set_trace() # 为什么有的
    return np.array(batch_data_mask_tok) # 将这样一个二维列表转成numpy, 转成以类别转换的id

def mm_group_mask(batch,type="self",pad=0): # 根据一系列特殊的整数id去生成mask
    length = batch.shape[1]
    lis = []
    if type=="self-sentence":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    if ele != 1000:copy[copy == 1000] = 0
                    copy[copy != ele] = 0
                    copy[copy == ele] = 1
                    #print("self copy",copy)
                '''
                if ele == 1000:
                    copy[copy != ele] = 1
                    copy[copy == ele] = 0
                '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()       
            mask = np.expand_dims(mask,0) # 0维度扩展一下，便于concat
            lis.append(mask)
        res = np.concatenate(tuple(lis)) # 把一个batch的concat起来
    elif type == "other-sentence":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if (ele>=900)and(ele<=963): copy = np.zeros(length)
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[(copy>=900) & (copy<=963)] = 0 
                    copy[copy==1000] = 0 # 问题部分无效  # 首先应该不看视觉部分 
                    copy[copy ==ele] = 0 # 不看自己这部分的句子
                    copy[copy!= 0] = 1   
                    '''
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type == "question":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele==1000:
                    copy[copy==0] = -1
                    copy[copy==1] = 0 ## 这种操作就是1000和
                    copy[copy==-1] = 1
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    else:
        return "error"
    return res

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    # 这里也只是实现多层的一个stack
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             /math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # 0为无效token
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MMGroupAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MMGroupAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3) # 分别用于QKV
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def get_mm_mask(self,src,pad=0):
        mask = mm_src_to_mask(src) # 这个统一都是torch.from_numpy来实现， src_to_mask怎么实现，是根据句子的结构以及token所属的性质，分配相同或者不同类别的整数
        self.src_mask_self = torch.from_numpy(mm_group_mask(mask,"self-sentence",pad).astype('uint8')).unsqueeze(1) 
        # 这个是每个句子内部做self_attention，因为head是独立的，所以送给每个head来做
        self.src_mask_between = torch.from_numpy(mm_group_mask(mask,"other-sentence",pad).astype('uint8')).unsqueeze(1)
        # 每个句子和除了自己意外的所有句子来做
        self.src_mask_question = torch.from_numpy(mm_group_mask(mask, "question", pad).astype('uint8')).unsqueeze(1)
        # 使用这个相当于question和题干做cross-attention
        self.src_mask_global = (torch.from_numpy(mask) != pad).unsqueeze(-2).unsqueeze(1) # 这个是获得长度的mask
        self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape) # expand是通过引用复制的方法，完成复制，在某个维度上完全是一样的
        self.final = torch.cat((self.src_mask_between.cuda(),self.src_mask_self.cuda(),self.src_mask_global.cuda(),self.src_mask_question.cuda()),1)
        return self.final.cuda()

    def forward(self, query, key, value, mask=None):
        #print("query",query,"\nkey",key,"\nvalue",value)
        "Implements Figure 2"
        mask = torch.cat((mask, mask), 1) # 重复2次就可以了
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k) # 再还原到某个维度
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MMFusionBlock(nn.Module):
    def __init__(self, cfg):
        super(MMFusionBlock, self).__init__()
        self.hidden_size = cfg.encoder_hidden_size

        N = cfg.group_attention_layers
        # add_some_parameter
        ff = PositionwiseFeedForward(cfg.encoder_hidden_size, cfg.d_ff_hidden, cfg.dropout_rate) # d_ff_hidden=2048
        self.cross_group_attention = MMGroupAttention(args.group_head_num, cfg.encoder_hidden_size,dropout=cfg.dropout_rate)
        self.cross_onelayer = Encoder(EncoderLayer(cfg.encoder_hidden_size,deepcopy(self.cross_group_attention), deepcopy(ff), cfg.dropout_rate),N) 

    def forward(self, src_emb, text_token_ids):
        src_mask = self.cross_group_attention.get_mm_mask(text_token_ids[:,0,:]) # (batch_size, batch_pad_len)
        mm_feature = self.cross_onelayer(src_emb, src_mask) 
        return mm_feature