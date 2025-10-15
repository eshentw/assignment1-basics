import os
import regex as re
import multiprocessing
import heapq
from collections import Counter
from typing import Dict, Tuple, List, Iterable, Iterator


def load_tokenizer(vocab_path: str, merges_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    vocab: Dict[int, bytes] = {}
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            if not line.strip():
                continue
            token_id_str, token_hex = line.rstrip().split("\t")
            vocab[int(token_id_str)] = bytes.fromhex(token_hex)
    merges: List[Tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as merges_file:
        for line in merges_file:
            if not line.strip():
                continue
            left_hex, right_hex = line.rstrip().split("\t")
            merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
    return vocab, merges


class Tokenizer():
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.vocab = dict(vocab) if vocab is not None else {}
        self.merges = list(merges) if merges is not None else []
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.special_tokens = list(special_tokens) if special_tokens else []
        token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._base_pre_token_pattern = token_pattern
        self._base_pre_token_re = re.compile(self._base_pre_token_pattern)
        self._special_prefixes: set[str] = set()
        self._special_prefix_lengths: List[int] = []
        if self.special_tokens:
            escaped_specials = [re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)]
            self._special_re = re.compile("|".join(escaped_specials))
            prefix_set: set[str] = set()
            for token in self.special_tokens:
                for i in range(1, len(token)):
                    prefix_set.add(token[:i])
            self._special_prefixes = prefix_set
            self._special_prefix_lengths = sorted({len(p) for p in prefix_set}, reverse=True)
        else:
            self._special_re = None
        
    def from_files(vocab_path: str, merges_path: str, special_tokens: List[str]):
        return Tokenizer(*load_tokenizer(vocab_path, merges_path), special_tokens)

    def _encode_pre_token(
                        self, 
                        pre_token_bytes: bytes, 
                        bpe_rank: Dict[Tuple[bytes, bytes], int]
                    ) -> List[int]:
        # Fast-path for complete tokens (e.g., special tokens) already present in vocab.
        token_id = self.token_to_id.get(pre_token_bytes)
        if token_id is not None:
            return [token_id]

        pre_token = tuple(bytes([b]) for b in pre_token_bytes)
        encoded: List[int] = []

        while len(pre_token) > 1:
            heap: List[Tuple[int, Tuple[bytes, bytes]]] = []
            for pair_bytes in zip(pre_token, pre_token[1:]):
                left, right = pair_bytes
                left_id = self.token_to_id.get(left)
                right_id = self.token_to_id.get(right)
                if left_id is None or right_id is None:
                    continue
                merged_pair = (self.vocab[left_id], self.vocab[right_id])
                rank = bpe_rank.get(merged_pair)
                if rank is not None:
                    heapq.heappush(heap, (rank, merged_pair))
            if not heap:
                break

            _, best_pair = heapq.heappop(heap)
            new_token = best_pair[0] + best_pair[1]
            new_pre_token: Tuple[bytes, ...] = tuple()
            i = 0
            while i < len(pre_token):
                if (
                    i < len(pre_token) - 1
                    and (pre_token[i], pre_token[i + 1]) == best_pair
                ):
                    new_pre_token += (new_token,)
                    i += 2
                else:
                    new_pre_token += (pre_token[i],)
                    i += 1
            pre_token = new_pre_token

        for token in pre_token:
            token_id = self.token_to_id.get(token)
            if token_id is None:
                raise ValueError(f"Token {token} not in vocabulary.")
            encoded.append(token_id)

        return encoded
        
    def encode(self, text: str) -> List[int]:
        return list(self.encode_iterable([text]))
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        if not self.vocab or not self.token_to_id:
            raise ValueError("Vocabulary is not initialized. Provide vocab and merges before encoding.")
        split_pat = "|".join(map(re.escape, self.special_tokens))
        bpe_rank = {pair: i for i, pair in enumerate(self.merges)}
        buffer = ""
        
        def consume_buffer(buf: str, flush: bool) -> Iterator[int]:
            idx = 0
            length = len(buf)
            guard_len = 0
            if not flush and self._special_prefix_lengths:
                for prefix_len in self._special_prefix_lengths:
                    if prefix_len <= length and buf[length - prefix_len:] in self._special_prefixes:
                        guard_len = prefix_len
                        break
            process_limit = length - guard_len
            if split_pat:
                chunks = [seg for seg in re.split(split_pat, buf[:process_limit])]
            else:
                chunks = [buf[:process_limit]]
            for chunk in chunks:
                if chunk:
                    for m in re.finditer(self._base_pre_token_pattern, chunk):
                        token_bytes = m.group(0).encode("utf-8")
                        for token_id in self._encode_pre_token(token_bytes, bpe_rank):
                            yield token_id
                idx += len(chunk)
                if self._special_re:
                    special_match = self._special_re.search(buf, idx, process_limit)
                    if special_match:
                        if not flush and special_match.end() == length:
                            return buf[special_match.start():]
                        token_bytes = special_match.group(0).encode("utf-8")
                        for token_id in self._encode_pre_token(token_bytes, bpe_rank):
                            yield token_id
                        idx = special_match.end()
            return buf[idx:]

        for fragment in texts:
            if not fragment:
                continue
            buffer += fragment
            buffer = yield from consume_buffer(buffer, flush=False) or ""
        buffer = yield from consume_buffer(buffer, flush=True) or ""
        if buffer:
            raise ValueError("Unprocessed text remained after encoding iterable.")
    
    def encode_iterable_stream(self, texts: Iterable[str]) -> Iterator[int]:
        if not self.vocab or not self.token_to_id:
            raise ValueError("Vocabulary is not initialized. Provide vocab and merges before encoding.")

        bpe_rank = {pair: i for i, pair in enumerate(self.merges)}
        buffer = ""

        def consume_segment(segment: str, flush_segment: bool) -> Iterator[int]:
            seg_idx = 0
            seg_len = len(segment)

            while seg_idx < seg_len:
                match = self._base_pre_token_re.match(segment, seg_idx)
                if not match:
                    if not flush_segment:
                        break
                    token_bytes = segment[seg_idx].encode("utf-8")
                    for token_id in self._encode_pre_token(token_bytes, bpe_rank):
                        yield token_id
                    seg_idx += 1
                    continue

                if not flush_segment and match.end() == seg_len:
                    break

                token_bytes = match.group(0).encode("utf-8")
                for token_id in self._encode_pre_token(token_bytes, bpe_rank):
                    yield token_id
                seg_idx = match.end()

            return segment[seg_idx:]

        def consume_buffer(buf: str, flush: bool) -> Iterator[int]:
            idx = 0
            length = len(buf)
            guard_len = 0
            # Don't split special tokens across buffer boundaries
            if not flush and self._special_prefix_lengths:
                for prefix_len in self._special_prefix_lengths:
                    if prefix_len <= length and buf[length - prefix_len:] in self._special_prefixes:
                        guard_len = prefix_len
                        break
            process_limit = length - guard_len

            while idx < process_limit:
                special_match = None
                # Deal with special tokens separately
                if self._special_re:
                    special_match = self._special_re.search(buf, idx, process_limit)
                next_cut = special_match.start() if special_match else length
                next_cut = min(next_cut, process_limit)
                if idx < next_cut:
                    segment_end = next_cut
                    segment = buf[idx:segment_end]
                    flush_segment = flush or (special_match is not None) or (process_limit < length)
                    segment_leftover = yield from consume_segment(segment, flush_segment=flush_segment)
                    if segment_leftover:
                        consumed = len(segment) - len(segment_leftover)
                        return buf[idx + consumed:]
                    idx = segment_end

                if special_match:
                    if not flush and special_match.end() == length:
                        return buf[special_match.start():]
                    token_bytes = special_match.group(0).encode("utf-8")
                    for token_id in self._encode_pre_token(token_bytes, bpe_rank):
                        yield token_id
                    idx = special_match.end()
                else:
                    break

            return buf[idx:]

        for fragment in texts:
            if not fragment:
                continue
            buffer += fragment
            buffer = yield from consume_buffer(buffer, flush=False) or ""

        buffer = yield from consume_buffer(buffer, flush=True) or ""
        if buffer:
            raise ValueError("Unprocessed text remained after encoding iterable.")
            
    def decode(self, token_ids: List[int]) -> str:
        if self.vocab is None:
            raise ValueError("Vocabulary is not initialized. Train the tokenizer first.")
        tokens = [self.vocab[token_id] for token_id in token_ids]
        return b"".join(tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab=None, merges=None, special_tokens=special_tokens)
    tokenizer.vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    tokenizer.token_to_id = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.vocab_size = len(tokenizer.vocab)
    tokenizer.merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    
    string = 'the cat ate'
    tokens = tokenizer.encode(string)
    print(f"Tokens for '{string}': {tokens}")
    assert tokens == [9, 7, 1, 5, 10, 3]
    

        
