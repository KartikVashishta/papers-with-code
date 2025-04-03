import os
import json
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset 
from tqdm import tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    tokens = [eot] 
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Add checkpoint tracking
CHECKPOINT_FILE = os.path.join(DATA_CACHE_DIR, "checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'shard_index': 0, 'processed_rows': 0}

def save_checkpoint(shard_index, processed_rows):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'shard_index': shard_index, 'processed_rows': processed_rows}, f)

# Load previous progress
checkpoint = load_checkpoint()
shard_index = checkpoint['shard_index']
skip_rows = checkpoint['processed_rows']

nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    # Skip already processed rows
    for i, tokens in enumerate(pool.imap(tokenize, fw.skip(skip_rows), chunksize=16)):
        try:
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
                
                # Save checkpoint after each shard
                save_checkpoint(shard_index, skip_rows + i + 1)
                
        except Exception as e:
            print(f"Error processing row {skip_rows + i}: {e}")
            save_checkpoint(shard_index, skip_rows + i)
            raise

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
        save_checkpoint(shard_index + 1, skip_rows + i + 1)