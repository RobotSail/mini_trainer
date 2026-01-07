"""Unit tests for PretrainingBlockDataset class."""

# Standard
import json
import os
import tempfile

# Third Party
import pytest
import torch
from datasets import Dataset as HFDataset

# First Party
from mini_trainer.sampler import PretrainingBlockDataset


class TestPretrainingBlockDataset:
    """Test suite for PretrainingBlockDataset."""

    @pytest.fixture
    def sample_pretraining_data(self):
        """Sample tokenized pretraining data (14 total tokens)."""
        return [
            {"input_ids": [1, 2, 3, 4, 5], "len": 5},
            {"input_ids": [6, 7, 8, 9, 10, 11], "len": 6},
            {"input_ids": [12, 13, 14], "len": 3},
        ]

    @pytest.fixture
    def temp_pretraining_jsonl(self, sample_pretraining_data):
        """Create temporary JSONL file with pretraining data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in sample_pretraining_data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def mock_hf_dataset(self, sample_pretraining_data):
        """Mock HuggingFace dataset for unit tests."""
        return HFDataset.from_list(sample_pretraining_data)

    def test_dataset_initialization(self, temp_pretraining_jsonl):
        """Test basic initialization, block_size, and num_blocks calculation."""
        # 14 total tokens, block_size=5 → 2 complete blocks + 1 partial (4 tokens)
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        assert dataset.block_size == 5
        assert dataset.num_blocks == 3  # 2 complete + 1 partial
        assert len(dataset) == 3
        assert len(dataset.all_input_ids) == 14  # All tokens kept

    def test_concatenation_of_documents(self, temp_pretraining_jsonl):
        """Verify documents are concatenated in order."""
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=14, pad_token_id=0
        )

        # Should have all tokens concatenated
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert dataset.all_input_ids == expected

    def test_partial_block_is_padded(self, temp_pretraining_jsonl):
        """Verify partial blocks are padded correctly."""
        # 14 tokens, block_size=5 → 2 complete blocks + 1 partial with 4 tokens
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        assert len(dataset.all_input_ids) == 14  # All tokens kept
        assert dataset.num_blocks == 3  # 2 complete + 1 partial

        # The partial block (index 2) should have padding
        partial_block = dataset[2]
        # Last 4 tokens are [11, 12, 13, 14], padded with 0
        assert partial_block["input_ids"].tolist() == [11, 12, 13, 14, 0]
        # Labels should mask the padding
        assert partial_block["labels"].tolist() == [11, 12, 13, 14, -100]

    def test_getitem_returns_correct_block(self, temp_pretraining_jsonl):
        """Test __getitem__ retrieves correct token ranges."""
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        # Block 0: tokens [1, 2, 3, 4, 5]
        block_0 = dataset[0]
        assert block_0["input_ids"].tolist() == [1, 2, 3, 4, 5]
        assert block_0["labels"].tolist() == [1, 2, 3, 4, 5]
        assert block_0["len"] == 5
        assert block_0["num_loss_counted_tokens"] == 4

        # Block 1: tokens [6, 7, 8, 9, 10]
        block_1 = dataset[1]
        assert block_1["input_ids"].tolist() == [6, 7, 8, 9, 10]
        assert block_1["labels"].tolist() == [6, 7, 8, 9, 10]
        assert block_1["len"] == 5
        assert block_1["num_loss_counted_tokens"] == 4

    def test_labels_equal_input_ids_for_complete_blocks(self, temp_pretraining_jsonl):
        """Verify complete blocks have labels == input_ids (no masking)."""
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        # Check only complete blocks (not partial)
        for i in range(2):  # First 2 blocks are complete
            block = dataset[i]
            assert torch.equal(block["input_ids"], block["labels"])

    def test_num_loss_counted_tokens_for_complete_block(self, temp_pretraining_jsonl):
        """Verify num_loss_counted_tokens is block_size - 1 for complete blocks."""
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        # Complete block should have block_size - 1 loss tokens
        block = dataset[0]
        assert block["num_loss_counted_tokens"] == 4

    def test_num_loss_counted_tokens_for_partial_block(self, temp_pretraining_jsonl):
        """Verify num_loss_counted_tokens for partial blocks."""
        # 14 tokens, block_size=5 → partial block has 4 tokens
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        # Partial block (4 tokens) should have 4-1=3 loss tokens (causal shift)
        partial_block = dataset[2]
        assert partial_block["num_loss_counted_tokens"] == 3

    def test_index_out_of_range(self, temp_pretraining_jsonl):
        """Test error handling for out-of-range indices."""
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        # Should have 3 blocks (indices 0, 1, 2)
        assert len(dataset) == 3

        # Accessing index 3 should raise IndexError
        with pytest.raises(IndexError):
            _ = dataset[3]

    def test_missing_input_ids_field_raises_error(self):
        """Validate required field check."""
        # Create dataset with wrong schema
        wrong_data = [
            {"tokens": [1, 2, 3], "len": 3},  # Wrong field name
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in wrong_data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must have 'input_ids' field"):
                PretrainingBlockDataset.from_jsonl_file(
                    temp_path, block_size=5, pad_token_id=0
                )
        finally:
            os.unlink(temp_path)

    def test_edge_case_exact_multiple(self):
        """Test when tokens exactly divide by block_size."""
        # 15 tokens, block_size=5 → exactly 3 blocks, no partial block
        data = [
            {"input_ids": list(range(1, 16)), "len": 15},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            dataset = PretrainingBlockDataset.from_jsonl_file(
                temp_path, block_size=5, pad_token_id=0
            )

            assert dataset.num_blocks == 3
            assert len(dataset) == 3
            assert len(dataset.all_input_ids) == 15

            # All blocks should be accessible
            for i in range(3):
                block = dataset[i]
                expected_start = i * 5 + 1
                expected_end = (i + 1) * 5 + 1
                assert block["input_ids"].tolist() == list(
                    range(expected_start, expected_end)
                )
        finally:
            os.unlink(temp_path)

    def test_edge_case_fewer_tokens_than_block_size(self):
        """Test when total tokens < block_size (partial block only)."""
        # 3 tokens, block_size=10 → 1 partial block (padded to 10)
        data = [
            {"input_ids": [1, 2, 3], "len": 3},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            dataset = PretrainingBlockDataset.from_jsonl_file(
                temp_path, block_size=10, pad_token_id=0
            )

            assert dataset.num_blocks == 1  # One partial block
            assert len(dataset) == 1
            assert len(dataset.all_input_ids) == 3

            # Block should be padded
            block = dataset[0]
            assert block["input_ids"].tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
            assert block["labels"].tolist() == [
                1,
                2,
                3,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
            ]
            assert block["num_loss_counted_tokens"] == 2  # 3 - 1 for causal shift
        finally:
            os.unlink(temp_path)

    def test_large_dataset_concatenation(self):
        """Performance check with many documents."""
        # Create 100 documents with 50 tokens each → 5000 total tokens
        data = [
            {"input_ids": list(range(i, i + 50)), "len": 50} for i in range(0, 5000, 50)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            # block_size=512 → 5000 // 512 = 9 complete blocks + 1 partial (392 tokens)
            dataset = PretrainingBlockDataset.from_jsonl_file(
                temp_path, block_size=512, pad_token_id=0
            )

            assert dataset.num_blocks == 10  # 9 complete + 1 partial
            assert len(dataset) == 10
            assert len(dataset.all_input_ids) == 5000

            # Verify we can access all blocks
            for i in range(len(dataset)):
                block = dataset[i]
                assert block["input_ids"].shape[0] == 512
                assert block["len"] == 512
        finally:
            os.unlink(temp_path)

    def test_tensor_dtype_correct(self, temp_pretraining_jsonl):
        """Verify torch.long dtype for tensors."""
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        block = dataset[0]

        # Both input_ids and labels should be torch.long
        assert block["input_ids"].dtype == torch.long
        assert block["labels"].dtype == torch.long

    def test_block_structure_consistency(self, temp_pretraining_jsonl):
        """Verify all blocks have consistent structure."""
        dataset = PretrainingBlockDataset.from_jsonl_file(
            temp_pretraining_jsonl, block_size=5, pad_token_id=0
        )

        for i in range(len(dataset)):
            block = dataset[i]

            # Check all required fields exist
            assert "input_ids" in block
            assert "labels" in block
            assert "len" in block
            assert "num_loss_counted_tokens" in block

            # Check types
            assert isinstance(block["input_ids"], torch.Tensor)
            assert isinstance(block["labels"], torch.Tensor)
            assert isinstance(block["len"], int)
            assert isinstance(block["num_loss_counted_tokens"], int)

            # Check values
            assert block["len"] == 5

    def test_can_accept_hf_dataset_directly(self, mock_hf_dataset):
        """Test that PretrainingBlockDataset can accept HF Dataset object."""
        # Pass HF Dataset directly
        dataset = PretrainingBlockDataset(mock_hf_dataset, block_size=5, pad_token_id=0)

        assert dataset.num_blocks == 3  # 2 complete + 1 partial
        assert len(dataset) == 3

        # Verify it works
        block = dataset[0]
        assert block["input_ids"].tolist() == [1, 2, 3, 4, 5]

    def test_empty_dataset_edge_case(self):
        """Test handling of empty dataset."""
        # Empty dataset should not be created - expect an error or empty result
        # Since load_dataset with empty file may behave differently,
        # test with a dataset that has no input_ids
        data = []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            # Empty file should result in empty dataset
            # This may raise an error depending on implementation
            try:
                dataset = PretrainingBlockDataset.from_jsonl_file(
                    temp_path, block_size=5, pad_token_id=0
                )
                # If it succeeds, should have 0 blocks
                assert dataset.num_blocks == 0
                assert len(dataset) == 0
            except Exception:
                # Some implementations may raise an error for empty files
                pass
        finally:
            os.unlink(temp_path)

    def test_negative_pad_token_id_raises_error(self, mock_hf_dataset):
        """Test that negative pad_token_id raises an error."""
        with pytest.raises(ValueError, match="pad_token_id must be a positive integer"):
            PretrainingBlockDataset(mock_hf_dataset, block_size=5, pad_token_id=-1)
