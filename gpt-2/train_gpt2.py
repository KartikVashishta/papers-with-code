from dataclasses import dataclass
from time import time
import torch
import math
import inspect
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # Number of tokens: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # batch_size, seq_len, embd_dim
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        # attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # the pos and the token embeddings must be forwarded
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # logits: B, T, vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2'          : dict(n_layer = 12, n_head=12, n_embd = 768),          
            'gpt2-medium'   : dict(n_layer = 24, n_head=16, n_embd = 1024),                  
            'gpt2-large'    : dict(n_layer = 36, n_head=20, n_embd = 1280),                 
            'gpt2-xl'       : dict(n_layer = 48, n_head=25, n_embd = 1600),              
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] 
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear layer.
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismached keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"number of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        return optimizer
         
# ------------------------------------------------------------
import tiktoken
import os
from pathlib import Path
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
    
        file_path = Path("data/tiny_shakespeare.txt")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = B * T * self.process_rank 

    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += B*T*self.num_processes
        
        if self.current_position + B*T*self.num_processes + 1 >= len(self.tokens):
            self.current_position = B * T * self.process_rank

        return x, y
# ------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for ddp, cuda is required"
    init_process_group(backend = 'nccl')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size must be divisible by B * T * ddp_world_size, but got {total_batch_size} and {B} * {T} * {ddp_world_size}"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Effective grad_accum_steps: {grad_accum_steps}")

train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it: int):
    # 1. Linear warmup for the first warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2. After warmup_steps, decay with cosine decay down to min learning rate
    if it > max_steps:
        return min_lr
    # 3. In between, use cosine decay down to min learning rate
    decay_ratio = (it-warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1..0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas = (0.9, 0.95), eps = 1e-8)
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)

for step in range(50):
    start_time = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):

        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type = device, dtype = torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_second = tokens_processed / (time.time() - start_time)
    if master_process:
        print(f"step {step:4d}, loss: {loss_accum.item():.5f}, lr: {lr:.6e}, norm: {norm:.4f}, tok/s: {tokens_per_second:4.0f}")

if ddp:
    destroy_process_group()

# model.eval()
# num_return_sequences = 5
# max_length = 30
# tokens = enc.encode("Hello, I'm a language model")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits, _ = model(x) # logits: B, T, vocab_size
#     logits = logits[:,-1,:] # logits: B, vocab_size

#     probs = F.softmax(logits, dim=-1)
#     # Top-k sampling
#     top_k = 50
#     topk_probs, topk_indices = torch.topk(probs, top_k, dim = -1)
#     # Select a token from the top-k
#     ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
#     xcol = torch.gather(topk_indices, dim=-1, index=ix) # (B, 1)
#     x = torch.cat((x, xcol), dim=1) # (B, T+1)

# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
