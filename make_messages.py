##
from datasets import load_dataset

data_path = "GAIR/LIMO"
output_path = "./limo_messages.jsonl"
dataset = load_dataset(data_path, split="train")

dataset
"""
shows:
    Out[2]: 
    Dataset({
        features: ['question', 'solution', 'answer'],
        num_rows: 817
    })
"""


##
def make_messages(sample: dict):
    x = sample["question"]
    y = sample["solution"]
    messages = [{"role": "user", "content": x}, {"role": "assistant", "content": y}]
    sample["messages"] = messages
    return sample


dataset = dataset.map(make_messages, num_proc=8)
dataset.to_json(output_path)
