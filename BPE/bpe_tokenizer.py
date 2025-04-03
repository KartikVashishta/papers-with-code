import os
import json
import regex as re
from collections import defaultdict, Counter, deque
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set, Union, Any

# Implementation inspired by concepts from Sebastian Raschka's BPE tutorial
# https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
class BPETokenizer:
    def __init__(self, vocab_size: int = 256):
        self.base_vocab_size = 256
        self.target_vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(self.base_vocab_size)}
        self.merges = {}
        self.pattern = None
        self.special_tokens = set()
        
    def _get_token_pairs(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge_pair(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, corpus: Union[str, List[str]], allowed_special: Set[str] = None, verbose: bool = False):
        if isinstance(corpus, str):
            corpus = [corpus]
        
        if allowed_special:
            self.special_tokens = set(allowed_special)
        
        # Preprocess text by replacing spaces with 'Ġ' (following GPT tokenizer convention)
        processed_corpus = []
        for text in corpus:
            processed_text = []
            for i, char in enumerate(text):
                if char == " " and i != 0:
                    processed_text.append("Ġ")
                elif char != " ":
                    processed_text.append(char)
            processed_corpus.append("".join(processed_text))
        
        all_tokens = []
        for text in processed_corpus:
            tokens = list(text.encode('utf-8'))
            all_tokens.extend(tokens)
        
        ids = list(all_tokens)
        n_merges = self.target_vocab_size - self.base_vocab_size - len(self.special_tokens)
        
        for i in range(n_merges):
            stats = self._get_token_pairs(ids)
            if not stats:
                break
                
            top_pair = max(stats, key=stats.get)
            idx = self.base_vocab_size + i
            
            if verbose:
                print(f"Merge #{i}: {top_pair} -> {idx} (occurs {stats[top_pair]} times)")
                
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            ids = self._merge_pair(ids, top_pair, idx)
        
        # Add special tokens after merges
        if self.special_tokens:
            for token in self.special_tokens:
                idx = len(self.vocab)
                token_bytes = token.encode('utf-8')
                self.vocab[idx] = token_bytes
        
        self._build_regex_pattern()
        return self
    
    def _build_regex_pattern(self):
        vocab_str = {i: token.decode('utf-8', errors='replace') for i, token in self.vocab.items()}
        
        sorted_vocab = sorted(vocab_str.items(), key=lambda x: len(x[1]), reverse=True)
        escaped_tokens = [re.escape(token) for _, token in sorted_vocab]
        self.pattern = re.compile('|'.join(escaped_tokens))
        
    def encode(self, text: str) -> List[int]:
        # Preprocess input text similar to training (replace spaces with 'Ġ')
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            elif char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)
        
        for token_id, token_bytes in self.vocab.items():
            if token_bytes.decode('utf-8', errors='replace') in self.special_tokens and token_bytes.decode('utf-8', errors='replace') in processed_text:
                # Handle special tokens as atomic units
                pass  # Placeholder for special token handling
    
        tokens = list(processed_text.encode('utf-8'))

        while len(tokens) >= 2:
            pairs = self._get_token_pairs(tokens)
            if not pairs:
                break
                
            # Find the pair with lowest merge index (highest priority)
            pair = min(pairs.keys(), key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
                
            idx = self.merges[pair]
            tokens = self._merge_pair(tokens, pair, idx)
            
        return tokens
    
    def tokenize_with_bpe(self, token: str) -> List[int]:
        # Initial character-level tokenization
        token_ids = [ord(char) for char in token if ord(char) < self.base_vocab_size]
        
        # Apply BPE merges
        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.merges:
                    merged_token_id = self.merges[pair]
                    new_tokens.append(merged_token_id)
                    i += 2
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens
            
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        tokens = b''.join([self.vocab[id] for id in ids])
        text = tokens.decode('utf-8', errors='replace')
        
        # Replace 'Ġ' with space during decoding
        text = text.replace('Ġ', ' ')
        return text
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.target_vocab_size,
                'merges': list(self.merges.items()),
                'special_tokens': list(self.special_tokens)
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.target_vocab_size = data['vocab_size']
        self.merges = {tuple(map(int, pair)): idx for pair, idx in data['merges']}
        self.special_tokens = set(data.get('special_tokens', []))
        
        # Rebuild vocab
        self.vocab = {i: bytes([i]) for i in range(self.base_vocab_size)}
        for (x, y), idx in self.merges.items():
            self.vocab[idx] = self.vocab[x] + self.vocab[y]
            
        # Add special tokens
        for token in self.special_tokens:
            idx = len(self.vocab)
            self.vocab[idx] = token.encode('utf-8')
        
        self._build_regex_pattern()
        return self
        
    def load_from_openai_format(self, encoder_path: str, vocab_path: str):
        """
        Load tokenizer from OpenAI's GPT format files.
        
        Args:
            encoder_path: Path to the encoder.json file
            vocab_path: Path to the vocab.bpe file
        """

        with open(encoder_path, 'r', encoding='utf-8') as f:
            encoder = json.load(f)
    
        inverse_encoder = {v: k for k, v in encoder.items()}
        max_id = max(int(v) for v in inverse_encoder.keys())
        
        self.vocab = {}
        for token_str, token_id in encoder.items():
            self.vocab[int(token_id)] = token_str.encode('utf-8')
            
        with open(vocab_path, 'r', encoding='utf-8') as f:
            bpe_merges = f.read().split('\n')

        if bpe_merges and bpe_merges[0].startswith('#'):
            bpe_merges = bpe_merges[1:]
            
        # Process merges
        self.merges = {}
        for i, merge in enumerate(bpe_merges):
            if not merge.strip():
                continue
                
            tokens = merge.split()
            if len(tokens) != 2:
                continue
                
            token1, token2 = tokens
            
            if token1 in encoder and token2 in encoder:
                token_id1 = int(encoder[token1])
                token_id2 = int(encoder[token2])
                merged_token = token1 + token2
                
                if merged_token in encoder:
                    merged_id = int(encoder[merged_token])
                    self.merges[(token_id1, token_id2)] = merged_id
                    
        self.target_vocab_size = len(self.vocab)
        self._build_regex_pattern()
        return self
    
    @lru_cache(maxsize=1024)
    def get_token_id(self, token: str) -> Optional[int]:
        """Get ID for a token string, with caching for performance."""
        token_bytes = token.encode('utf-8')
        for id, bytes_val in self.vocab.items():
            if bytes_val == token_bytes:
                return id
        return None


if __name__ == "__main__":
    
    text = "Hello in Japanese is said as, Konnichiwa: こんにちは"
    
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(text)
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {text == decoded}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Compression ratio: {len(text)/len(encoded):.2f}x")