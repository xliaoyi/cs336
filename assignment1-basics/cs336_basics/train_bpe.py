
import regex as re
import os
import pickle
import time

from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO, List, Tuple, Dict
from functools import partial
from tqdm import tqdm


def process_chunk(text, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    escaped = "|".join(re.escape(t) for t in special_tokens)
    parts = re.split(escaped, text)

    counter = Counter()
    for part in parts:
        for t in re.finditer(PAT, part):
            m = t.group().encode("utf-8")
            counter[tuple(bytes([b]) for b in m)] += 1
            # counter[m] += 1
    return counter


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


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # split text
    chunks = []
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # print(start, end)
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    
    # with Pool(num_processes) as pool:
    #     counters = pool.map(process_chunk, chunks)
    with Pool(num_processes) as pool:
        counters = pool.map(
            partial(process_chunk, special_tokens=special_tokens),
            chunks,
        )


    # merge all counters
    all_counter = Counter()
    for c in counters:
        all_counter.update(c)
    
    token_states = list(all_counter.keys())
    token_counts = list(all_counter.values())

    # train bpe
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    next_id = 256

    for t in special_tokens:
        vocab[next_id] = t.encode('utf-8')
        next_id += 1
        
    merges = []

    pair_to_tokens = defaultdict(set)
    pair_counts = Counter()
    for token_id, (token, count) in enumerate(all_counter.items()):
        for j in range(len(token) - 1):
            pair_to_tokens[(token[j], token[j + 1])].add(token_id)
            pair_counts[(token[j], token[j + 1])] += count

    # tqdm
    pbar = tqdm(total=vocab_size - next_id, desc="Learning BPE merges")
    while next_id < vocab_size:
        max_count = max(pair_counts.values())
        keys = [k for k, v in pair_counts.items() if v == max_count]
        max_key = max(keys)

        vocab[next_id] = max_key[0] + max_key[1]
        merges.append(max_key)
        next_id += 1

        token_ids = pair_to_tokens[max_key].copy()
        for token_id in token_ids:
            token = token_states[token_id]

            # remove adjacent pair
            for pair in zip(token[:-1], token[1:]):
                pair_counts[pair] -= token_counts[token_id]
                if pair_counts[pair] == 0:
                    pair_counts.pop(pair)
            for pair in set(zip(token[:-1], token[1:])):    
                pair_to_tokens[pair].remove(token_id)
                if pair_to_tokens[pair] == set():
                    pair_to_tokens.pop(pair)

            new_token = []
            j = 0
            while j < len(token):
                if j < len(token) - 1 and (token[j], token[j + 1]) == max_key:
                    new_token.append(token[j] + token[j + 1])
                    j += 2
                else:
                    new_token.append(token[j])
                    j += 1
            token_states[token_id] = tuple(new_token)

            for j in range(len(new_token) - 1):
                pair_to_tokens[(new_token[j], new_token[j + 1])].add(token_id)
                pair_counts[(new_token[j], new_token[j + 1])] += token_counts[token_id]

        pbar.update(1)
    pbar.close()

    return vocab, merges

if __name__ == "__main__":

    import json

    # tiny stories
    print("**training BPE for tiny stories")
    vocab, merges = train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=['<|endoftext|>']
    )

    json_vocab = {
        str(k): list(v)
        for k, v in vocab.items()
    }

    json_merges = [
        [list(a), list(b)]
        for a, b in merges
    ]

    with open("./tiny_stories_vocab.json", "w") as f:
        json.dump(json_vocab, f)

    with open("./tiny_stories_merges.json", "w") as f:
        json.dump(json_merges, f)

    print(f"tiny stories vocab saved to ./tiny_stories_merges.json")
    print(f"tiny stories merges saved to ./tiny_stories_merges.json")
    print()

    # open web text
    print("**training BPE for open web text")
    vocab, merges = train_bpe(
        input_path="../data/owt_train.txt",
        vocab_size=32000,
        special_tokens=['<|endoftext|>']
    )

    json_vocab = {
        str(k): list(v)
        for k, v in vocab.items()
    }

    json_merges = [
        [list(a), list(b)]
        for a, b in merges
    ]

    with open("./owt_vocab.json", "w") as f:
        json.dump(json_vocab, f)

    with open("./owt_merges.json", "w") as f:
        json.dump(json_merges, f)

    print(f"open web text vocab saved to ./tiny_stories_merges.json")
    print(f"open web text merges saved to ./tiny_stories_merges.json")
    print()