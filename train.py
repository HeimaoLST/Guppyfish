from datasets import load_dataset

from tokenizer import Tokenizer

ds = load_dataset("arman-bd/guppylm-60k-generic")


def process_data(examples, max_length=64):
    tokenizer = Tokenizer("tokenizer.json")
    all_texts = []
    all_input_ids = []
    batch_size = len(examples["input"])
    for batch_index in range(batch_size):
        raw_text = (
            "<|user|> "
            + examples["input"][batch_index]
            + " <|assistant|> "
            + examples["output"][batch_index]
        )

        ids = tokenizer.encode(raw_text)
        padded_ids = []
        if len(ids) > max_length:
            padded_ids = ids[:max_length]
        else:
            pad_id = tokenizer.vocab.get("<pad>", 20)
            padded_ids = ids + [pad_id] * (max_length - len(ids))
        all_texts.append(raw_text)
        all_input_ids.append(padded_ids)
    return {"input_ids": all_input_ids, "text": all_texts}


ds = ds["train"].map(process_data, batched=True)


print(ds[0]["text"])
print(ds[0]["input_ids"])
#
#
