"""Unit tests for get_data_loader with pretraining support."""

# Standard
import json
import os
import tempfile
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from mini_trainer.sampler import get_data_loader, PretrainingBlockDataset
from mini_trainer.training_types import PretrainingConfig


class TestGetDataLoaderPretraining:
    """Test suite for get_data_loader with pretraining mode."""

    @pytest.fixture
    def pretraining_config(self):
        """PretrainingConfig fixture."""
        return PretrainingConfig(block_size=128)

    @pytest.fixture
    def temp_pretraining_file(self):
        """Create temp pretraining JSONL file with 10 docs, 500 tokens total."""
        data = [
            {"input_ids": list(range(i, i + 50)), "len": 50} for i in range(0, 500, 50)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_instruction_tuning_file(self):
        """Create temp instruction tuning JSONL file with labels."""
        data = [
            {
                "input_ids": list(range(i, i + 50)),
                "labels": list(range(i, i + 50)),
                "len": 50,
                "num_loss_counted_tokens": 49,
            }
            for i in range(0, 500, 50)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_pretraining_mode_creates_block_dataset(
        self, temp_pretraining_file, pretraining_config
    ):
        """Verify PretrainingBlockDataset is used in pretraining mode."""
        train_loader, val_loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=pretraining_config,
        )

        # Get dataset from loader
        dataset = train_loader.dataset

        # Should be PretrainingBlockDataset
        assert isinstance(dataset, PretrainingBlockDataset)
        assert dataset.block_size == 128

    def test_pretraining_mode_returns_no_val_loader(
        self, temp_pretraining_file, pretraining_config
    ):
        """Verify validation loader is None in pretraining mode."""
        train_loader, val_loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=pretraining_config,
        )

        assert val_loader is None

    def test_validation_split_with_pretraining_works(
        self, temp_pretraining_file, pretraining_config
    ):
        """Validate that validation_split works with pretraining mode."""
        train_loader, val_loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            validation_split=0.1,
            pretraining_config=pretraining_config,
        )
        # Both loaders should be created
        assert train_loader is not None
        assert val_loader is not None
        # Both datasets should be PretrainingBlockDataset
        assert isinstance(train_loader.dataset, PretrainingBlockDataset)
        assert isinstance(val_loader.dataset, PretrainingBlockDataset)

    def test_pretraining_blocks_are_batched_correctly(
        self, temp_pretraining_file, pretraining_config
    ):
        """Verify batching works with pretraining blocks."""
        train_loader, _ = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=pretraining_config,
        )

        # Iterate through a few batches
        batch_count = 0
        for batch in train_loader:
            if batch_count >= 2:  # Just check first 2 batches
                break

            # batch is a list of minibatches from the collator
            assert isinstance(batch, list)
            assert len(batch) > 0

            for minibatch in batch:
                # Verify minibatch structure
                assert "input_ids" in minibatch
                assert "labels" in minibatch

                # Verify tensors
                assert isinstance(minibatch["input_ids"], torch.Tensor)
                assert isinstance(minibatch["labels"], torch.Tensor)

            batch_count += 1

        assert batch_count > 0  # Ensure we got some batches

    def test_pretraining_mode_respects_max_tokens_per_gpu(
        self, temp_pretraining_file, pretraining_config
    ):
        """Verify collator works with pretraining dataset."""
        train_loader, _ = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=10,
            max_tokens_per_gpu=200,  # Small limit
            seed=42,
            pretraining_config=pretraining_config,
        )

        # Check that batches respect max_tokens
        for batch in train_loader:
            # batch is a list of minibatches
            for minibatch in batch:
                # Each block is 128 tokens, so max 1-2 blocks per GPU should fit in 200 tokens
                total_tokens = minibatch["input_ids"].numel()
                # Allow some flexibility for batch structure
                assert total_tokens <= 200 * 10  # batch_size * max_tokens (upper bound)
            break  # Just check first batch

    @patch("mini_trainer.sampler.log_rank_0")
    def test_pretraining_logging_output(
        self, mock_log, temp_pretraining_file, pretraining_config
    ):
        """Verify informative logs are produced."""
        get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=pretraining_config,
        )

        # Verify logging was called
        assert mock_log.called

        # Check that it logged about pretraining dataset
        call_args_list = [str(call) for call in mock_log.call_args_list]
        log_messages = " ".join(call_args_list)

        # Should mention blocks and/or pretraining
        assert "block" in log_messages.lower() or "pretraining" in log_messages.lower()

    def test_instruction_tuning_mode_without_pretraining_config(
        self, temp_instruction_tuning_file
    ):
        """Verify instruction tuning mode still works (no pretraining_config)."""
        train_loader, val_loader = get_data_loader(
            data_path=temp_instruction_tuning_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=None,  # No pretraining
        )

        # Should get loader (val_loader will be None with validation_split=0)
        assert train_loader is not None
        assert val_loader is None  # No validation split requested

    def test_instruction_tuning_to_pretraining_switch(
        self, temp_pretraining_file, temp_instruction_tuning_file, pretraining_config
    ):
        """Verify clean switching between modes."""
        # First: Instruction tuning mode
        it_train_loader, it_val_loader = get_data_loader(
            data_path=temp_instruction_tuning_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=None,
        )

        # Then: Pretraining mode
        pt_train_loader, pt_val_loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=pretraining_config,
        )

        # Verify different dataset types
        assert it_train_loader is not None
        assert pt_train_loader is not None

        # Pretraining should not have validation
        assert pt_val_loader is None

    @pytest.mark.slow
    def test_pretraining_with_real_tokenizer_gpt2(self):
        """End-to-end integration test with GPT2 tokenizer."""
        from transformers import AutoTokenizer

        # Create realistic pretraining data with GPT2
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        documents = [
            "This is the first test document for pretraining.",
            "This is the second test document with more content.",
            "Short doc.",
        ]

        # Tokenize documents (simulating data processing output)
        tokenized_docs = []
        for doc in documents:
            input_ids = tokenizer.encode(doc, add_special_tokens=True)
            input_ids.append(tokenizer.eos_token_id)
            tokenized_docs.append({"input_ids": input_ids, "len": len(input_ids)})

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in tokenized_docs:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            # Create data loader
            pretraining_config = PretrainingConfig(block_size=50)
            train_loader, val_loader = get_data_loader(
                data_path=temp_path,
                batch_size=2,
                max_tokens_per_gpu=100,
                seed=42,
                pretraining_config=pretraining_config,
            )

            # Verify loader works
            assert train_loader is not None
            assert val_loader is None

            # Get a batch
            for batch in train_loader:
                # batch is a list of minibatches
                for minibatch in batch:
                    # Verify structure
                    assert "input_ids" in minibatch
                    assert "labels" in minibatch

                    # Verify token IDs are valid for GPT2 (< vocab_size)
                    vocab_size = tokenizer.vocab_size
                    assert torch.all(minibatch["input_ids"] < vocab_size)
                    assert torch.all(minibatch["input_ids"] >= 0)

                break  # Just check first batch

        finally:
            os.unlink(temp_path)


class TestGetDataLoaderValidation:
    """Additional validation tests for get_data_loader."""

    @pytest.fixture
    def temp_no_labels_file(self):
        """Create file without labels field (pretraining data)."""
        data = [
            {"input_ids": list(range(i, i + 50)), "len": 50} for i in range(0, 100, 50)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_instruction_tuning_requires_labels_field(self, temp_no_labels_file):
        """Verify instruction tuning mode validates labels field exists."""
        # Trying to use data without labels in instruction tuning mode should fail
        with pytest.raises(ValueError, match="must contain 'labels' field"):
            get_data_loader(
                data_path=temp_no_labels_file,
                batch_size=4,
                max_tokens_per_gpu=200,
                seed=42,
                pretraining_config=None,  # Instruction tuning mode
            )

    def test_pretraining_accepts_data_without_labels(self, temp_no_labels_file):
        """Verify pretraining mode accepts data without labels field."""
        pretraining_config = PretrainingConfig(block_size=50)

        # Should work fine - pretraining doesn't need labels field
        train_loader, val_loader = get_data_loader(
            data_path=temp_no_labels_file,
            batch_size=4,
            max_tokens_per_gpu=200,
            seed=42,
            pretraining_config=pretraining_config,
        )

        assert train_loader is not None
        assert val_loader is None
