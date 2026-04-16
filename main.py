import json
import re

from datasets import load_dataset

ds = load_dataset("arman-bd/guppylm-60k-generic")







def print_dict(dict: dict):
    print("len:", len(dict))
    print(list(dict.items())[0:20])


def pretoken(word_dict: dict, text: str):

    words = re.findall(r"\w+'\w+|\w+|[.!?,]", text)

    for word in words:
        word_dict[word] = word_dict.get(word, 0) + 1

    # word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
    return word_dict


def toVocab(word_dict: dict):
    return {" ".join(list(word)) + " </w>": freq for word, freq in word_dict.items()}


def textToSplit(text, merges):

    words = re.findall(r"\w+'\w+|\w+|[,.!?]", text)
    result = []
    # split = [" ".join(list(word)) + " </w>" for word in words]
    for word in words:
        split = " ".join(list(word)) + " </w>"
        for pair in merges:
            bigp = " ".join(pair)
            replacement = "".join(pair)
            split = split.replace(bigp, replacement)
        result.append(split)
    return result







def build_token_set(vocab: dict):
    tokens_set = set()
    for word in vocab.keys():
        tokens_set.update(word.split())
    return tokens_set


def encode(tokenizer: Tokenizer, text):
    clean_text = preprocess(text)
    split = textToSplit(clean_text, tokenizer.merges)
    result = []
    for word in split:
        result.append(tokenizer.vocab.get(word, "<unk>"))
    return result



def BPEtrainer(word_dict: dict, aimDemisons: int):
    vocab = toVocab(word_dict)
    tokens_set = build_token_set(vocab)
    merges = []
    maxpair = ("", "")
    while len(tokens_set) < aimDemisons:
        maxpair = findMaxPair(vocab)
        if maxpair == ("", ""):
            break

        merges.append(maxpair)
        vocab = mergeVocab(vocab, maxpair)
        tokens_set = build_token_set(vocab)

    # print_dict(vocab)
    print(len(tokens_set))
    return tokens_set, merges


def main():

    train = ds["train"]
    word_dict = {}
    for row in train:
        # row = train[i]
        iotext = (
            preprocess(row["input"]) + " " + preprocess(row["output"])  # type: ignore
        )  # ignore: type
        word_dict = pretoken(word_dict, iotext)

    # print_dict(word_dict)

    token_set, merges = BPEtrainer(word_dict, 1024)

    save_tokenizer(token_set, merges)
    test = ds["test"]
    test_text = test[0]["input"] + " " + test[0]["output"]
    tokenizer = Tokenizer("tokenizer.json")
    encoding = encode(tokenizer, test_text)
    print(encoding)


if __name__ == "__main__":
    main()
