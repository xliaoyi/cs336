import json
import regex as re
import json
import numpy as np

from tqdm import tqdm
from typing import BinaryIO, List, Tuple, Dict, Iterable


class Tokenizer:
	def __init__(self, vocab, merges, special_tokens = None):
		self.vocab = vocab
		self.vocab_inverse = {v: k for k, v in self.vocab.items()}
		self.merges = merges
		if special_tokens is None:
			self.special_tokens = []
		else:
			self.special_tokens = special_tokens
	
	@classmethod
	def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
		# Load vocab
		with open(vocab_filepath, "r") as f:
			json_vocab = json.load(f)

		vocab = {
			int(k): bytes(v)
			for k, v in json_vocab.items()
		}

		# Load merges
		with open(merges_filepath, "r") as f:
			json_merges = json.load(f)

		merges = [
			(bytes(a), bytes(b))
			for a, b in json_merges
			]
		
		# handle new special tokens
		if special_tokens is not None:
			next_id = max(vocab.keys()) + 1
			for token in special_tokens:
				token_bytes = token.encode('utf-8')
				if token_bytes not in vocab.values():
					vocab[next_id] = token_bytes
					next_id += 1
		
		return cls(vocab, merges, special_tokens)

	def encode(self, text):
		# split text based on special tokens
		# escaped = "|".join(re.escape(t) for t in self.special_tokens)
		if self.special_tokens:
			escaped = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
			parts = re.split(f"({escaped})", text)
		else:
			parts = [text]

		PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
		
		text_token_list = []
		for part in parts:
			if part == "":
				continue

			if part in self.special_tokens:
				text_token_list.append([part.encode('utf-8')])
				continue
			
			for t in re.finditer(PAT, part):
				m = t.group().encode("utf-8")
				text_token_list.append([bytes([b]) for b in m])

		# check if text_token_list is 2d list (split by special token or not)
		if len(text_token_list) == 0 or not isinstance(text_token_list[0], list):
			text_token_list = [text_token_list]

		# merge -- apply bpe
		for tu in self.merges:
			merged_text_token_list = []
			for part in text_token_list:
				i = 0
				merged_part = []
				while i < len(part):
					if i < len(part) - 1 and (part[i], part[i + 1]) == tu:
						merged_part.append(part[i] + part[i + 1])
						i += 2
					else:
						merged_part.append(part[i])
						i += 1
				merged_text_token_list.append(merged_part)
			text_token_list = merged_text_token_list
		
		# flatten
		text_token_list = [x for row in text_token_list for x in row]

		text_id_list = [self.vocab_inverse[token] for token in text_token_list]
		return text_id_list


	def encode_iterable(self, iterable: Iterable[str]):
		for line in iterable:
			for token_id in self.encode(line):
				yield token_id

	def decode(self, ids: list[int]):
		text = b''
		for i in ids:
			text += self.vocab[i]
		return text.decode('utf-8', errors="replace")


if __name__ == "__main__":

	# tiny stories
	tokenizer = Tokenizer.from_files(
		vocab_filepath = "tiny_stories_vocab.json",
		merges_filepath = "tiny_stories_merges.json",
		special_tokens = ["<|endoftext|>"]
	)

	
	with open("../data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
		ids = np.fromiter(tokenizer.encode_iterable(tqdm(f, desc="Tokenizing")), dtype=np.uint16)

	np.save("tiny_stories_train.npy", ids)

	with open("../data/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
		ids = np.fromiter(tokenizer.encode_iterable(tqdm(f, desc="Tokenizing")), dtype=np.uint16)

	np.save("tiny_stories_valid.npy", ids)

	# owt
	# tokenizer = Tokenizer.from_files(
	# 	vocab_path = "owt_vocab.json",
	# 	merges_path = "owt_merges.json",
	# 	special_tokens = ["<|endoftext|>"]
	# )

	
	# with open("../data/owt_train.txt") as f:
	# 	ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

	# np.save("owt_train.npy", ids)