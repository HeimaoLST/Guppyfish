import json
import re

from train_tokenizer import merge_symbols, preprocess


class Tokenizer:
    vocab: dict
    merges: list

    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [tuple(merge) for merge in data["merges"]]


def encode(tokenizer: Tokenizer, text):
    words = re.findall(r"\w+'\w+|\w+|[,.!?]", text)
    symbols = [" ".join(list(word)) + " </w>" for word in words]
    new_list = []
    for word in symbols:
        for pair in tokenizer.merges:
            word = merge_symbols(word, pair)
        new_list.append(word)
    result = []

    for word in new_list:
        result.extend(word.split())

    ids = []
    print("tokenizer token: ", result)
    for word in result:
        ids.append(tokenizer.vocab[word])
    return ids


text = "a snail could be nice. they're slow so they won't eat my food fast. snail i cleaned your tankwater is good right now. the current feels gentle. water is life for me."

tokenizer = Tokenizer("tokenizer.json")
print(encode(tokenizer, preprocess(text)))
