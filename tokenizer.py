import json
import re

from train_tokenizer import merge_symbols, preprocess, split_text


class Tokenizer:
    vocab: dict
    merges: list
    id2token: dict

    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [tuple(merge) for merge in data["merges"]]
        self.id2token = data["id2token"]


def encode(tokenizer: Tokenizer, text):
    # words = re.findall(r"\w+'\w+|\w+|[,.!?]", text)
    words = split_text(text)
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


def decode(tokenizer: Tokenizer, ids):
    words = []
    for id in ids:
        words.append(tokenizer.id2token.get(str(id)))
    text = "".join(words)
    text = text.replace("</w>", " ")
    return text


text = "a snail could be nice. they're slow so they won't eat my food fast. snail i cleaned your tankwater is good right now. the current feels gentle. water is life for me."

tokenizer = Tokenizer("tokenizer.json")
print("raw text: \n", text)
ids = encode(tokenizer, preprocess(text))
print("encode: \n", ids)

print("decode: \n", decode(tokenizer, ids))
