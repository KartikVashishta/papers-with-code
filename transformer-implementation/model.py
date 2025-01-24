import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(Q, K, V, mask = None, dropout: nn.Dropout = None):
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, V), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):

        super().__init__()
        # d_model should be divisible by h
        assert d_model % h == 0
      
        self.d_k = d_model // h
        self.h = h

        # Q, K, V and output linear layers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None):

        batch_size = Q.size(0) 
        # Q, K, V -> [batch_size, seq_len, d_model] -> [batch_size, seq_len, h, d_k]
        # Convert Q, K, V from [batch_size, seq_len, h, d_k] -> [batch_size, h, seq_len, d_k]
        Q, K, V = [
            lin(x).view(batch_size,-1,self.h, self.d_k).transpose(1,2) for lin, x in zip(self.linears, (Q, K, V))
        ]
        # self.attn -> weights
        # x -> output
        x, self.attn = attention(Q, K, V, mask = mask, dropout = self.dropout) 
        # x -> [batch_size, h, seq_len, d_k]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):

        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, attn: MultiHeadedAttention, ffn: PositionwiseFFN, dropout: float = 0.1):

        super().__init__()
        self.self_attn = attn
        self.ffn = ffn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        x = x + self.dropout(self.self_attn(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, attn: MultiHeadedAttention, ffn: PositionwiseFFN, dropout: float = 0.1):

        super().__init__()
        self.self_attn = copy.deepcopy(attn)
        self.src_attn = copy.deepcopy(attn)
        self.ffn = ffn

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(x, x, x, tgt_mask))
        x = self.norm1(x)
        x = x + self.dropout(self.src_attn(x, memory, memory, src_mask))
        x = self.norm2(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm3(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Embedding lookup + scaling
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        # 1/10000^(2i/d_model) => e^[log(10000) * (-2i/d_model)]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
class Transformer(nn.Module):
   
    def __init__(self, src_vocab: int, 
                 tgt_vocab: int, 
                 d_model : int = 512, 
                 N : int = 6, 
                 d_ff : int = 2048, 
                 h: int = 8, 
                 dropout : float = 0.1):

        super().__init__()

        self_attn = MultiHeadedAttention(h, d_model)
        src_attn = MultiHeadedAttention(h, d_model)

        ff = PositionwiseFFN(d_model, d_ff, dropout)
        
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, copy.deepcopy(self_attn), copy.deepcopy(ff), dropout)
            for _ in range(N)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, copy.deepcopy(self_attn), copy.deepcopy(src_attn), copy.deepcopy(ff), dropout)
            for _ in range(N)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab),
            PositionalEncoding(d_model, dropout)
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(d_model, tgt_vocab),
            PositionalEncoding(d_model, dropout)
        )

        self.final_layer = nn.Linear(d_model, tgt_vocab)

        # if the src_vocab == tgt_vocab, we can use the same embedding, i.e, weight tying
        if src_vocab == tgt_vocab:
            self.final_layer.weight = self.tgt_embed[0].lut.weight
            self.src_embed[0].lut.weight = self.tgt_embed[0].lut.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
       
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.encoder_norm(x)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        
        x = self.tgt_embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.decoder_norm(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
       
        memory = self.encode(src, src_mask)
        x = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.final_layer(x)
    
def create_mask(src, tgt, pad_idx = 0):
    
    # src_mask: [batch_size, 1, 1, seq_len] => True if not padding
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    # tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] => True if not padding
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # No-peek mask => create an upper triangular matrix of 1s for allowed pos
    seq_len = tgt.size(1)
    no_peek_mask = ~(torch.tril(torch.ones(seq_len, seq_len)).bool())
    tgt_mask = tgt_mask & no_peek_mask.to(src.device)
    return src_mask, tgt_mask

class LabelSmoothing(nn.Module):
    def __init__(self, size: int, padding_idx: int = 0, smoothing: float = 0.0):
       
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction = 'sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        # true_dist: [batch_size, size]
        true_dist = x.data.clone()
        # Fill everything with smoothing / (size - 2), except for the target and <pad>
        true_dist.fill_(self.smoothing / (self.size - 2))
        # Fill the target with confidence
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # Fill the padding with 0
        true_dist[:, self.padding_idx] = 0
        # Mask out padding tokens
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)
    
class WarmupScheduler:
    def __init__(self, d_model: int, warmup_steps: int):
        
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def get_lr(self):
        # lr = d_model ^ (-0.5) * min(step_num ^ (-0.5), step_num * warmup_steps ^ (-1.5))
        return (
            self.d_model ** (-0.5) * min(
                (self.step_num+1) ** (-0.5),
                (self.step_num+1) * self.warmup_steps ** (-1.5)
            )
        )

    def step(self):
        self.step_num += 1

