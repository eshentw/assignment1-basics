import os
from datasets import load_dataset
import multiprocessing
import regex as re
import heapq
from collections import Counter
from typing import Dict, Tuple, BinaryIO, List


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def heap_key(pair: Tuple[int, int], vocab: Dict[int, bytes]) -> tuple[int, ...]:
    left = vocab[pair[0]]
    right = vocab[pair[1]]
    left_key = tuple(-byte for byte in left)
    right_key = tuple(-byte for byte in right)
    return left_key + (-len(left),) + right_key + (-len(right),)

# def heap_key(pair: Tuple[int, int], vocab: Dict[int, bytes]) -> tuple[int, ...]:
# This  implementation lost the boundary between left and right tokens — so:
# Two different pairs (e.g., (A,B) and (AA,) if those exist) might produce the same combined byte sequence.
# There’s no tie-breaker for pairs of the same combined bytes but different lengths.
# Heap ordering becomes unstable or inconsistent between runs (depending on Python’s tuple comparison fallback).
#     left = vocab[pair[0]]
#     right = vocab[pair[1]]
#     combined = left + right
#     return tuple(-byte for byte in combined)

class BPETrainer:
    def __init__(
        self,):
        self.pre_token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.corpus_map: Dict[Tuple[bytes, ...], int] = {}
        self.vocab_size = 0
        self.max_vocab_size = 0
        self.special_tokens: list[str] = []
        self.vocab: Dict[int, bytes] = {}
        self.token_to_id: Dict[bytes, int] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        
    def __initialize(self, data_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = 4):
        self.special_tokens = special_tokens
        self.merges: List[Tuple[bytes, bytes]] = []
        self.vocab, self.token_to_id = None, None
        self.build_vocab(special_tokens)
        assert self.vocab is not None and self.token_to_id is not None
        self.max_vocab_size = vocab_size
        split_special = special_tokens[0].encode("utf-8") if special_tokens else b""
        if data_path is not None:
            self.corpus_map = self.processing_corpus(
                data_path, num_workers, split_special, special_tokens
            )
        else:
            raise ValueError("Data path must be provided for training.")
            
    def build_vocab(self, special_tokens) -> None:
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.token_to_id: Dict[bytes, int] = {token: token_id for token_id, token in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab:
                self.token_to_id[token_bytes] = self.vocab_size
                self.vocab[self.vocab_size] = token_bytes
                self.vocab_size += 1
    
    def processing_corpus(
            self, file_name, n_proc, split_special_token, special_tokens
        ) -> Dict[Tuple[bytes, bytes], int]:
        with open(file_name, "rb") as file:
            boundaries = find_chunk_boundaries(file, n_proc, split_special_token)

        if n_proc <= 1:
            total_token_freq = Counter()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                total_token_freq.update(self.pre_tokenize_worker(file_name, start, end, special_tokens))
        else:
            with multiprocessing.Pool(n_proc) as pool:
                results = [
                    pool.apply_async(self.pre_tokenize_worker, args=(file_name, start, end, special_tokens))
                    for start, end in zip(boundaries[:-1], boundaries[1:])
                ]
                token_freqs = [res.get() for res in results]
                total_token_freq = Counter()
                for tf in token_freqs:
                    total_token_freq.update(tf)
        return dict(total_token_freq)

    def pre_tokenize_worker(
                    self, 
                    file_name: str, 
                    start: int, 
                    end: int, 
                    special_tokens: list[str],
                ) -> Counter:
        with open(file_name, "rb") as file:
            file.seek(start)
            corpus_chunk = file.read(end - start).decode('utf-8', errors='ignore')
        return self.pre_tokenize_text(corpus_chunk, special_tokens)

    def pre_tokenize_text(
                    self,
                    corpus_chunk: str,
                    special_tokens: list[str],
                ) -> Dict[Tuple[bytes, bytes], int]:
        token_freq:Dict[Tuple[bytes, bytes], int] = Counter()
        # Remove special token from the chunk if present
        # and split the chunk by the special token
        split_pat = "|".join(map(re.escape, special_tokens))
        corpus_chunks = [seg for seg in re.split(split_pat, corpus_chunk) if seg]
        for corpus_chunk in corpus_chunks:
            for m in re.finditer(self.pre_token_pattern, corpus_chunk):
                token = tuple(m.group(0).encode('utf-8'))
                token_freq[token] += 1
        return token_freq
    
    def get_pair(self, ) -> Dict[Tuple[bytes, bytes], int]:
        pair_freq: Dict[Tuple[bytes, bytes], int] = Counter()
        for seq, freq in self.corpus_map.items():
            if len(seq) < 2:
                continue
            for pair in zip(seq, seq[1:]):
                pair_freq[pair] += freq
        return pair_freq

    def train(
            self,
            file_path: str | None = None,
            vocab_size: int | None = None,
            special_tokens: list[str] | None = None,
            num_workers: int = 4,
        ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        '''
        Return:
            vocab - The new vocab mapping from token id to token bytes.
            merges - The list of merges performed, in order, as pairs of token bytes.
        '''
        if file_path is not None:
            if special_tokens is None:
                special_tokens = self.special_tokens or ["<|endoftext|>"]
            if vocab_size is None:
                vocab_size = self.max_vocab_size
            self.__initialize(file_path, vocab_size, special_tokens, num_workers)
        elif self.vocab is None or self.token_to_id is None:
            raise ValueError("Tokenizer is uninitialized. Provide training data or load weights before calling train.")

        assert self.vocab is not None and self.token_to_id is not None 
        assert self.corpus_map is not None

        pair_freq = self.get_pair()

        if not pair_freq:
            return self.vocab, self.merges

        heap = [(-freq, heap_key(pair, self.vocab), pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)
        self.merges = []

        def decrement_pair(pair: Tuple[int, int], amount: int):
            current = pair_freq.get(pair)
            if current is None:
                return
            new_val = current - amount
            if new_val <= 0:
                pair_freq.pop(pair, None)
            else:
                pair_freq[pair] = new_val
                heapq.heappush(heap, (-new_val, heap_key(pair, self.vocab), pair))

        def increment_pair(pair: Tuple[int, int], amount: int):
            new_val = pair_freq.get(pair, 0) + amount
            pair_freq[pair] = new_val
            heapq.heappush(heap, (-new_val, heap_key(pair, self.vocab), pair))

        while self.vocab_size < self.max_vocab_size:
            while heap:
                neg_freq, _, pair = heapq.heappop(heap)
                freq = -neg_freq
                current = pair_freq.get(pair)
                if current is None:
                    continue
                if current != freq:
                    heapq.heappush(heap, (-current, heap_key(pair, self.vocab), pair))
                    continue
                best_pair = pair
                break
            else:
                break

            best_count = pair_freq.get(best_pair, 0)
            if best_count <= 0:
                break

            left_bytes, right_bytes = self.vocab[best_pair[0]], self.vocab[best_pair[1]]
            new_token = left_bytes + right_bytes
            new_id = self.vocab_size
            self.vocab[new_id] = new_token
            self.token_to_id[new_token] = new_id
            self.vocab_size += 1
            self.merges.append((left_bytes, right_bytes))
            pair_freq.pop(best_pair, None)
            new_freq_map: Dict[Tuple[bytes, bytes], int] = {}
            any_sequence_updated = False

            for seq_tuple, seq_freq in self.corpus_map.items():
                seq = list(seq_tuple)
                if len(seq) < 2:
                    new_freq_map[seq_tuple] = new_freq_map.get(seq_tuple, 0) + seq_freq
                    continue

                merged: list[int] = []
                idx = 0
                updated = False

                while idx < len(seq):
                    if (
                        idx + 1 < len(seq)
                        and self.vocab[seq[idx]] == left_bytes
                        and self.vocab[seq[idx + 1]] == right_bytes
                    ):  
                        if not updated:
                            for a, b in zip(seq, seq[1:]):
                                decrement_pair((a, b), seq_freq)
                            updated = True
                            any_sequence_updated = True
                        merged.append(new_id)
                        idx += 2
                    else:
                        merged.append(seq[idx])
                        idx += 1

                merged_tuple = tuple(merged)
                new_freq_map[merged_tuple] = new_freq_map.get(merged_tuple, 0) + seq_freq

                if updated and len(merged) >= 2:
                    for a, b in zip(merged, merged[1:]):
                        increment_pair((a, b), seq_freq)

            if not any_sequence_updated:
                # No corpus updates; revert merge and treat as convergence
                self.vocab.pop(new_id, None)
                self.token_to_id.pop(new_token, None)
                self.vocab_size -= 1
                self.merges.pop()
                break

            self.corpus_map = new_freq_map

        return self.vocab, self.merges
    
    def save(self, vocab_path: str, merges_path: str) -> None:
        if self.vocab is None:
            raise ValueError("Vocabulary is not initialized. Train the tokenizer first.")
        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir:
            os.makedirs(vocab_dir, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            for token_id, token_bytes in sorted(self.vocab.items()):
                vocab_file.write(f"{token_id}\t{token_bytes.hex()}\n")
        merges_dir = os.path.dirname(merges_path)
        if merges_dir:
            os.makedirs(merges_dir, exist_ok=True)
        with open(merges_path, "w", encoding="utf-8") as merges_file:
            for left_bytes, right_bytes in self.merges:
                merges_file.write(f"{left_bytes.hex()}\t{right_bytes.hex()}\n")


def train_bpe(file_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = 4):
    tokenizer = BPETrainer()
    tokenizer.train(
        file_path=file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_workers=num_workers,
    )
    return tokenizer


if __name__ == "__main__":
    import time
    # file_name = "/home/eddie880509/src/llm_from_scratch/assignment1-basics/cs336_basics/text_corpus.txt"
    weight_path = "/home/eddie880509/src/llm_from_scratch/assignment1-basics/tinystory"
    n_proc = 1
    special_tokens = ["<|endoftext|>"]
    # tokenizer = BPETrainer()
    # vocab, merges = tokenizer.train(file_path=file_name, vocab_size=10000, special_tokens=special_tokens, num_workers=n_proc)
    # find the longest token in the vocab
    # assert tokenizer.vocab is not None
    # longest_token = max(tokenizer.vocab.values(), key=len)
    # print(f"Longest token length: {len(longest_token)}")
    # print(f"Longest token bytes: {longest_token}")
    