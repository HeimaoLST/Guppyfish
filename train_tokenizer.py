import json
import re

from datasets import Dataset, load_dataset

ds = load_dataset("arman-bd/guppylm-60k-generic")


def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def toVocab(word_dict: dict):
    return {" ".join(list(word)) + " </w>": freq for word, freq in word_dict.items()}


def split_text(text: str):
    return re.findall(r"\w+'\w+|\w+|[,.!?]", text.lower())


def split_word(word: str):
    return word.split()


def build_token_set(vocab: dict):
    tokens_set = set()
    for word in vocab.keys():
        tokens_set.update(word.split())
    return tokens_set


def findMaxPair(vocab: dict):
    from collections import defaultdict

    freq_dict = defaultdict(int)
    maxpair = ("", "")
    maxfreq = 0

    for word, freq in vocab.items():
        symbol = split_word(word)
        for i in range(len(symbol) - 1):
            freq_dict[(symbol[i], symbol[i + 1])] += freq
            if freq_dict[(symbol[i], symbol[i + 1])] > maxfreq:
                maxfreq = freq_dict[(symbol[i], symbol[i + 1])]
                maxpair = (symbol[i], symbol[i + 1])
    return maxpair


def merge_symbols(word: str, pair: tuple):
    merged = []
    symbols = split_word(word)
    i = 0
    while i < len(symbols):
        if (i < len(symbols) - 1) and ((symbols[i], symbols[i + 1]) == pair):
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return " ".join(merged)


def mergeVocab(vocab: dict, pair: tuple):
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = merge_symbols(word, pair)
        new_vocab[new_word] = new_vocab.get(new_word, 0) + freq
    return new_vocab


def BPEtrainer(dataset: Dataset, aimDemisons: int):
    from collections import defaultdict

    word_dict = defaultdict(int)
    for row in dataset:
        iotext = row["input"] + " " + row["output"]
        iotext = preprocess(iotext)
        words = split_text(iotext)
        for word in words:
            word_dict[word] += 1

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
        new_token = "".join(maxpair)
        tokens_set.add(new_token)

    # print_dict(vocab)
    print(len(tokens_set))
    return tokens_set, merges


def save_tokenizer(token_set: set, merges, path="tokenizer.json"):
    special_token = ["<unk>"]
    token_set.update(special_token)
    vocab = {token: i for i, token in enumerate(sorted(token_set))}
    id2token = {i: token for i, token in enumerate(sorted(token_set))}
    data = {
        "vocab": vocab,
        "merges": [list(word) for word in merges],
        "id2token": id2token,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {len(vocab)} tokens, {len(merges)} merges")


def train_tokenizer(dataset: Dataset, vocab_size):
    token_set, merges = BPEtrainer(dataset, vocab_size)
    save_tokenizer(token_set, merges)


def main():
    train = ds["train"]
    train_tokenizer(train, 1024)


main()
