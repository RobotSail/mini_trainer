"""
This script implements a custom data loading and batching pipeline specifically
designed for efficient distributed training of sequence models, particularly
large language models, on multiple GPUs.

Key Features:
- Epoch-based Sampler: Provides shuffled data indices for each epoch,
  suitable for both finite and infinite training modes.
- Initial Batching: Groups samples into initial batches based on a fixed number
  of samples per batch.
- Dynamic Minibatching for Distributed Training: Takes the initial batches and
  further divides them into 'minibatches'. Each minibatch is a list distributed
  across available ranks (GPUs). The allocation process aims to pack sequences
  efficiently such that the total number of tokens processed by any single rank
  within a minibatch step stays below a predefined maximum (`max_tokens_per_gpu`).
  The number of minibatches generated from an initial batch can vary dynamically
  depending on the lengths of the sequences in that batch.
- Token-Based Load Balancing: Ensures that each GPU receives a comparable
  computational load (measured in tokens) per step, optimizing hardware
  utilization and preventing out-of-memory errors when dealing with variable
  sequence lengths.
- Padding/Dummy Samples: Handles cases where ranks might not have enough data
  to fill a minibatch by using dummy samples, ensuring all ranks process the
  same number of minibatches.
"""
from deprecated import deprecated
import os

import torch
from torch.utils.data import Sampler, Dataset, DataLoader, SequentialSampler
import torch.distributed as dist
import numpy as np
from datasets import load_dataset, Dataset as HFDataset
from mini_trainer.batch_packer import batch_lengths_to_minibatches_lpt
from mini_trainer.utils import log_rank_0

def reset_minibatches(num_ranks: int):
    return [[] for _ in range(num_ranks)], np.zeros(num_ranks)



@deprecated("Use batch_lengths_to_minibatches_lpt instead for better load balancing performance")
def batch_lengths_to_minibatches(batch_lengths: list[int], max_tokens_per_rank: int, num_ranks: int, rank: int):
    """Distributes indices from a batch into minibatches across ranks.

    Takes a list of sequence lengths corresponding to samples in an initial batch
    and distributes their indices into multiple 'minibatches'. Each minibatch
    represents a step where data is processed concurrently across `num_ranks` GPUs.

    The distribution aims to assign sequences (represented by their indices `sid`
    in the original `batch_lengths` list) to ranks such that the sum of sequence
    lengths (tokens) assigned to any single rank does not exceed
    `max_tokens_per_rank`. It prioritizes assigning the next sequence to the rank
    currently having the minimum total tokens assigned in the current minibatch.

    If adding the next sequence to the least-loaded rank would exceed the limit,
    the current minibatch is considered complete, and a new minibatch is started.

    If the last minibatch is incomplete, ranks with no assigned sequences are
    given a placeholder index of -1.

    Args:
        batch_lengths: A list where each element is the length (number of tokens)
                       of a sequence in the initial batch.
        max_tokens_per_rank: The maximum number of tokens allowed per rank in a
                             single minibatch.
        num_ranks: The total number of distributed training ranks (GPUs).
        rank: The specific rank for which to retrieve the assigned indices.

    Returns:
        A list of lists. Each inner list contains the indices (from the original
        batch) assigned to the specified `rank` for one minibatch. Placeholder -1
        indicates padding.
    """
    minibatches_indices = []
    current_minibatches_ids, current_minibatches_loads = reset_minibatches(num_ranks)
    for sid, sample_len in enumerate(batch_lengths):
        least_full_batch_id = np.argmin(current_minibatches_loads)
        
        if current_minibatches_loads[least_full_batch_id] + sample_len > max_tokens_per_rank:
            '''when the least full minibatch is full, we need to start a new minibatch'''
            minibatches_indices.append(current_minibatches_ids)
            current_minibatches_ids, current_minibatches_loads = reset_minibatches(num_ranks)
            least_full_batch_id = 0
        
        '''add sample to the least full minibatch'''
        current_minibatches_ids[least_full_batch_id].append(sid)
        current_minibatches_loads[least_full_batch_id] += sample_len
    
    if any(current_minibatches_loads):
        for i in range(num_ranks):
            if current_minibatches_loads[i] == 0:
                current_minibatches_ids[i].append(-1)
        minibatches_indices.append(current_minibatches_ids)
        
    return [m[rank] for m in minibatches_indices]

class JsonlDataset(Dataset):
    def __init__(
        self,
        path: str | None = None,
        max_seq_len: int | None = None,
        hf_dataset: HFDataset | None = None, 
    ):
        """
        Initializes a JsonlDataset object which we use to load and process the dataset.
        Accepts either a path to a JSONL file or a pre-loaded HuggingFace dataset.

        Args:
            path: Path to the JSONL file or HuggingFace dataset name
            max_seq_len: Maximum sequence length to keep (filters out longer sequences)
            hf_dataset: Pre-loaded HuggingFace dataset
        """
        # dataset can be any of these
        if hf_dataset is not None:
            dataset = hf_dataset
        elif path is not None:
            dataset = load_dataset("json", data_files=path, split="train")
        else:
            raise ValueError("Either 'path' or 'hf_dataset' must be provided")

        # The two required fields on a dataset are `input_ids` and `labels`,
        # everything else is computable. Here we handle the case when we
        # must actually provide them
        dataset = self.add_necessary_fields(dataset)
        if max_seq_len is not None:
            dataset = self.filter_by_max_seq_len(dataset, max_seq_len)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[int(index)]
        
        # Determine the number of loss-counted tokens if the field is missing.
        if (loss_counted_tokens := sample.get("num_loss_counted_tokens", None)) is None:
            loss_counted_tokens = sum(
                1 if label != -100 else 0 for label in sample["labels"]
            )

        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
            'len': sample['len'],
            'num_loss_counted_tokens': loss_counted_tokens,
        }
    
    @classmethod
    def add_necessary_fields(cls, dataset: HFDataset) -> HFDataset:
        required_fields = ["input_ids", "labels"]
        for field in required_fields:
            if field not in dataset.features:
                raise ValueError(f"Dataset must contain '{field}' field")
        if "len" not in dataset.features:
            dataset = dataset.map(lambda s: {"len": len(s["input_ids"])})
        if "num_loss_counted_tokens" not in dataset.features:
            dataset = dataset.map(
                lambda s: {
                    "num_loss_counted_tokens": sum(
                        1 for tok in s["labels"] if tok != -100
                    )
                }
            )
        
        return dataset

    @classmethod
    def filter_by_max_seq_len(cls, dataset: HFDataset, max_seq_len: int) -> HFDataset:
        dataset = dataset.filter(lambda x: x["len"] <= max_seq_len)
        return dataset
    
    @classmethod
    def load_and_split(
        cls, 
        data_path: str, 
        validation_split: float = 0.0,
        max_seq_len: int | None = None,
        seed: int = 42
    ) -> tuple["JsonlDataset", "JsonlDataset | None"]:
        """Load dataset and optionally split into train/validation sets.
        
        Args:
            data_path: Path to JSONL file or HuggingFace dataset name
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
            max_seq_len: Maximum sequence length (filters out longer sequences)
            seed: Random seed for reproducible splits
            
        Returns:
            tuple: (train_dataset, val_dataset) where val_dataset is None if validation_split <= 0
        """
        # handle either local or HF dataset
        if os.path.exists(data_path):
            hf_dataset = load_dataset("json", data_files=data_path, split="train")
        else:
            hf_dataset = load_dataset(data_path, split="train")
        
        # add necessary fields & filter by max_seq_len if specified
        hf_dataset = cls.add_necessary_fields(hf_dataset)
        if max_seq_len is not None:
            original_size = len(hf_dataset)
            hf_dataset = cls.filter_by_max_seq_len(hf_dataset, max_seq_len)
            filtered_size = len(hf_dataset)
            if original_size > filtered_size:
                log_rank_0(
                    f"\033[33mFiltered out {original_size - filtered_size} samples "
                    f"(out of {original_size}) that exceed max_seq_len={max_seq_len}\033[0m"
                )
        
        val_dataset = None
        if validation_split <= 0.0:
            # default case
            train_dataset = cls(hf_dataset=hf_dataset)
            return train_dataset, val_dataset
        
        # validation split case
        split_dataset = hf_dataset.train_test_split(
            test_size=validation_split,
            seed=seed,
            shuffle=True
        )
        train_dataset = cls(hf_dataset=split_dataset["train"])
        val_dataset = cls(hf_dataset=split_dataset["test"])
        return train_dataset, val_dataset
    
class EpochSampler(Sampler):
    """
    Here we redefine RandomSampler so we can have a consistent signature with InfiniteSampler
    """
    def __init__(self, len_data: int, seed: int = 67, epoch: int = 0):
        self.len_data = len_data
        self.seed = seed
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def generate_samples(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        samples = torch.randperm(self.len_data, generator=g).tolist()
        return samples 

    def __iter__(self):
        samples = self.generate_samples()
        yield from samples

    def __len__(self):
        return self.len_data


def mb_collate_fn(minibatch, batch_num_loss_counted_tokens):
    """Collates a list of samples into a single packed batch for Flash Attention.

    This function takes a 'minibatch' (list of pre-fetched dataset samples)
    and concatenates their 'input_ids', 'labels', and generates corresponding
    'position_ids'. It does *not* add padding.

    The resulting batch format is 'packed' or 'unpadded', where multiple sequences
    are concatenated into single tensors. Sequence boundaries are implicitly defined
    by the 'position_ids', which restart from 0 for each concatenated sequence.

    **IMPORTANT**: This format requires the downstream model's attention mechanism
    (e.g., Flash Attention) to correctly handle packed sequences. Standard attention
    implementations may not work correctly as they expect padded inputs and explicit
    attention masks. Flash Attention typically uses mechanisms like `cu_seqlens`
    (cumulative sequence lengths), derived from position IDs or sequence lengths,
    to compute the correct block-diagonal attention implicitly.

    Args:
        minibatch: A list of dictionaries, where each dictionary represents a
                   sample and contains at least 'input_ids' and 'labels'.

    Returns:
        A dictionary containing the collated batch:
        - 'input_ids': Single tensor of concatenated input IDs.
        - 'labels': Single tensor of concatenated labels.
        - 'position_ids': Single tensor of position IDs, reset for each sequence.
        - 'num_loss_counted_tokens': Total number of non-ignored label tokens (-100).
        - 'num_samples': The number of sequences packed into this batch.
    """
    input_ids = []
    labels = []
    position_ids = []
    total_len = 0
    num_loss_counted_tokens = 0
    # from ipdb import set_trace; set_trace()
    # try:
    num_samples = 0
    for item in minibatch:
        item_len = len(item["input_ids"])

        input_ids.extend(item["input_ids"])
        labels.extend(item["labels"])
        position_ids.extend(range(item_len))

        total_len += item_len
        # sample_loss_counted_tokens = (item["labels"] != -100).sum().item()
        num_loss_counted_tokens += item["num_loss_counted_tokens"]
        
        '''dummy samples don't have labels != -100 and should not count'''
        num_samples += 1 if item["num_loss_counted_tokens"] > 0 else 0 

    # print(
    #     f"\033[96m total length: {total_len} "
    #     f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
    # )

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "num_samples": num_samples,
        "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
    }
    
class MaxTokensPerRankCollator:
    """A collate function for PyTorch DataLoader for distributed training.

    This collator takes a batch of samples (obtained using indices from a sampler
    like InfiniteSampler) and performs two main tasks:
    1. Filters out samples longer than `max_tokens_per_rank`.
    2. Uses `batch_lengths_to_minibatches_lpt` to determine how to distribute the
       remaining samples across ranks into one or more 'minibatches', ensuring
       no rank exceeds `max_tokens_per_rank` per minibatch.
    3. For the current rank, it fetches the assigned samples (or dummy samples
       for padding) for each determined minibatch.
    4. Uses `mb_collate_fn` to collate the samples for each minibatch into the
       packed format required by Flash Attention.

    Args:
        max_tokens_per_rank (int): Maximum number of tokens allowed per rank
            in a single processed minibatch.
        rank (int, optional): The rank of the current process. If None, attempts
            to get it from `torch.distributed`.
        world_size (int, optional): Total number of ranks. If None, attempts
            to get it from `torch.distributed`.
        dummy_sample (dict, optional): A sample used for padding when a rank
            has no real samples assigned in a minibatch.
    """
    def __init__(self, max_tokens_per_rank: int, rank: int=None, world_size: int=None, dummy_sample=None):
        self.max_tokens_per_rank = max_tokens_per_rank

        if rank is None:
            self.global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        else:
            self.global_rank = rank
        if world_size is None:
            self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        else:
            self.world_size = world_size
        if dummy_sample is None:
            dummy_sample = {'input_ids': torch.tensor([15, 14, 13, 12, 11], dtype=torch.long),
                            'labels': torch.tensor([-100, -100, -100, -100, -100], dtype=torch.long),
                            'len': 5,
                            'num_loss_counted_tokens': 0}
        self.dummy_sample = dummy_sample

    def __call__(self, batch: list[dict]):
        """Processes a batch of samples into a list of packed minibatches for the current rank.

        Args:
            batch: A list of sample dictionaries from the Dataset.

        Returns:
            A list where each element is a dictionary representing a collated minibatch
            (output of `mb_collate_fn`) ready for processing by the current rank.
        """
        batch_ = [b for b in batch if b['len'] <= self.max_tokens_per_rank]
        if len(batch_) < len(batch):
            log_rank_0(f"\033[38;5;196mremoved {len(batch) - len(batch_)} samples from batch because they are longer than the max tokens per gpu\033[0m")
        # Use filtered batch for lengths and loss counts
        batch_lengths = [sample["len"] for sample in batch_]
        batch_num_loss_counted_tokens = sum(
            [sample["num_loss_counted_tokens"] for sample in batch_]
        )
        all_minibatches_indices = batch_lengths_to_minibatches_lpt(
            batch_lengths, self.max_tokens_per_rank, self.world_size, self.global_rank
        )

        all_minibatches = []
        for mb_indices in all_minibatches_indices:
            mb = [batch_[i] if i != -1 else self.dummy_sample for i in mb_indices]
            all_minibatches.append(mb_collate_fn(mb, batch_num_loss_counted_tokens))

        return all_minibatches


def get_data_loader(
    data_path: str,
    batch_size: int,
    max_tokens_per_gpu: int,
    seed: int,
    rank: int | None = None,
    world_size: int | None = None,
    dummy_sample: dict | None = None,
    num_workers: int = 0,
    validation_split: float = 0.0,
    max_seq_len: int | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """Create data loader(s) with optional train/validation split.
    
    Efficiently loads the dataset once and splits it if needed, avoiding
    multiple reads of the same data.
    
    Args:
        data_path: Path to the JSONL data file or HuggingFace dataset
        batch_size: Number of samples per batch
        max_tokens_per_gpu: Maximum tokens per GPU per step
        seed: Random seed for reproducibility
        rank: Rank of the current process (for distributed training)
        world_size: Total number of processes (for distributed training)
        dummy_sample: Sample to use for padding when ranks have uneven data
        num_workers: Number of worker processes for data loading
        validation_split: Fraction of data to use for validation (0.0 to 1.0)
        max_seq_len: Maximum sequence length to keep (filters out longer sequences)
    
    Returns:
        tuple: (train_loader, val_loader) where val_loader is None if validation_split <= 0
    """
    # validate validation_split parameter
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError(f"validation_split must be between 0 and 1 (exclusive of 1), got {validation_split}")
    
    # create the jsonl dataset and optionally the validation dataset
    train_dataset, val_dataset = JsonlDataset.load_and_split(
        data_path=data_path,
        validation_split=validation_split,
        max_seq_len=max_seq_len,
        seed=seed
    )
    if val_dataset is not None:
        log_rank_0(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    else:
        log_rank_0(f"Dataset split: {len(train_dataset)} train")

    
    # Create collate function
    collate_fn = MaxTokensPerRankCollator(
        max_tokens_per_gpu, 
        rank=rank, 
        world_size=world_size, 
        dummy_sample=dummy_sample,
    )
    
    # Create train data loader
    train_sampler = EpochSampler(len(train_dataset), seed=seed)
    train_loader = DataLoader(
        train_dataset, 
        batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )
    
    # Create validation data loader if needed
    val_loader = None
    if val_dataset is not None:
        val_sampler = SequentialSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset, 
            batch_size, 
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=False,
        )
    
    return train_loader, val_loader

if __name__ == "__main__":
    data_loader, _ = get_data_loader(data_path="test.jsonl",
                                     batch_size=40,
                                     max_tokens_per_gpu=5000,
                                     seed=37,
                                     rank=0,
                                     world_size=2)
    data_loader2, _ = get_data_loader(data_path="test.jsonl",
                                      batch_size=26,
                                      max_tokens_per_gpu=5000,
                                      seed=37,
                                      rank=1,
                                      world_size=2)
    data_loader = iter(data_loader)
    data_loader2 = iter(data_loader2)
    batch = next(data_loader)
    batch2 = next(data_loader2)
    from IPython import embed
    embed()

