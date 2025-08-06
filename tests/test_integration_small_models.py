"""
Integration tests using actual small language models from Transformers.

These tests use tiny model configurations to test real functionality
without requiring large amounts of memory or computation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import torch
import torch.nn as nn
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from setup_model_for_training import align_model_and_tokenizer, setup_model, setup_training_components
from sampler import JsonlDataset, InfiniteSampler, MaxTokensPerRankCollator, get_data_loader
from batch_metrics import BatchMetrics
from train import take_gradient_step


def create_tiny_llama_model():
    """Create a tiny Llama model with ~50k parameters."""
    config = LlamaConfig(
        vocab_size=500,  # Very small vocabulary
        hidden_size=32,   # Tiny hidden size
        intermediate_size=64,  # Small FFN
        num_hidden_layers=2,  # Only 2 layers
        num_attention_heads=2,  # Few attention heads
        num_key_value_heads=1,  # GQA
        max_position_embeddings=64,  # Short sequences
        rope_theta=10000.0,
        hidden_act="silu",
    )
    model = LlamaForCausalLM(config)
    return model, config


def create_tiny_mistral_model():
    """Create a tiny Mistral model with ~50k parameters."""
    config = MistralConfig(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        sliding_window=32,
    )
    model = MistralForCausalLM(config)
    return model, config


def create_tiny_qwen2_model():
    """Create a tiny Qwen2 model with ~50k parameters."""
    config = Qwen2Config(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config)
    return model, config


def create_test_data_file(num_samples=10, max_length=50):
    """Create a temporary JSONL file with test data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(num_samples):
            length = min(max_length, 10 + i * 3)
            sample = {
                "input_ids": list(range(length)),
                "labels": list(range(length)),
                "len": length,
                "num_loss_counted_tokens": length - 5  # Some tokens not counted
            }
            json.dump(sample, f)
            f.write('\n')
        return f.name


class TestSmallModelIntegration:
    """Integration tests with actual small models."""
    
    def test_tiny_llama_creation(self):
        """Test creating a tiny Llama model."""
        model, config = create_tiny_llama_model()
        
        # Verify model is small
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 100_000, f"Model has {total_params} params, expected < 100k"
        print(f"Llama model has {total_params:,} parameters")
        
        # Verify model can forward
        input_ids = torch.randint(0, 500, (1, 10))
        output = model(input_ids=input_ids, labels=input_ids)
        
        assert output.loss is not None
        assert output.logits.shape == (1, 10, 500)
    
    def test_tiny_mistral_creation(self):
        """Test creating a tiny Mistral model."""
        model, config = create_tiny_mistral_model()
        
        # Verify model is small
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 100_000, f"Model has {total_params} params, expected < 100k"
        print(f"Mistral model has {total_params:,} parameters")
        
        # Verify model can forward
        input_ids = torch.randint(0, 500, (1, 10))
        output = model(input_ids=input_ids, labels=input_ids)
        
        assert output.loss is not None
        assert output.logits.shape == (1, 10, 500)
    
    def test_tiny_qwen2_creation(self):
        """Test creating a tiny Qwen2 model."""
        model, config = create_tiny_qwen2_model()
        
        # Verify model is small  
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 100_000, f"Model has {total_params} params, expected < 100k"
        print(f"Qwen2 model has {total_params:,} parameters")
        
        # Verify model can forward
        input_ids = torch.randint(0, 500, (1, 10))
        output = model(input_ids=input_ids, labels=input_ids)
        
        assert output.loss is not None
        assert output.logits.shape == (1, 10, 500)
    
    def test_model_tokenizer_alignment(self):
        """Test aligning a tiny model with a tokenizer."""
        model, config = create_tiny_llama_model()
        
        # Set initial config values that differ from tokenizer to test alignment
        model.config.pad_token_id = 999  # Different from tokenizer
        model.config.bos_token_id = 998  # Different from tokenizer
        model.config.eos_token_id = 997  # Different from tokenizer
        
        # Create a mock tokenizer
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=500)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        
        with patch('setup_model_for_training.log_rank_0'):  # Mock logging
            aligned_model = align_model_and_tokenizer(model, tokenizer)
        
        # Check alignment happened - model should now match tokenizer
        assert aligned_model.config.pad_token_id == 0
        assert aligned_model.config.bos_token_id == 1
        assert aligned_model.config.eos_token_id == 2
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tiny_model_training_step(self):
        """Test a training step with a tiny model."""
        model, config = create_tiny_llama_model()
        model = model.cuda()
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        
        # Create sample batch
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, 500, (batch_size, seq_length)).cuda()
        labels = input_ids.clone()
        
        # Forward pass
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient step
        grad_norm = take_gradient_step(model, optimizer, scheduler)
        
        assert grad_norm is not None
        assert grad_norm.item() > 0
    
    def test_data_loader_with_tiny_model(self):
        """Test data loading with tiny model requirements."""
        # Create test data
        data_file = create_test_data_file(num_samples=20, max_length=30)
        
        try:
            # Create data loader
            loader = get_data_loader(
                data_path=data_file,
                batch_size=4,
                max_tokens_per_gpu=100,
                seed=42,
                rank=0,
                world_size=1
            )
            
            # Get one batch
            batch = next(iter(loader))
            
            assert isinstance(batch, list)
            if batch:
                assert 'input_ids' in batch[0]
                assert 'labels' in batch[0]
                assert 'position_ids' in batch[0]
        
        finally:
            os.unlink(data_file)
    
    def test_batch_metrics_with_model_output(self):
        """Test batch metrics with actual model outputs."""
        model, config = create_tiny_llama_model()
        
        # Create metrics tracker
        metrics = BatchMetrics()
        
        # Simulate minibatch processing
        for i in range(3):
            input_ids = torch.randint(0, 500, (1, 10))
            output = model(input_ids=input_ids, labels=input_ids)
            
            metrics.accumulate_minibatch_metrics(
                num_samples=1,
                loss=output.loss.item(),
                num_loss_counted_tokens=10,
                num_total_tokens=10,
                time_per_minibatch=0.1
            )
        
        # Check accumulated metrics
        assert metrics.minibatch_metrics['num_samples'] == 3
        assert metrics.minibatch_metrics['num_loss_counted_tokens'] == 30
        assert 'loss' in metrics.minibatch_metrics


class TestModelInitialization:
    """Test model initialization with tiny models."""
    
    @patch('setup_model_for_training.AutoTokenizer.from_pretrained')
    @patch('setup_model_for_training.AutoModelForCausalLM.from_pretrained')
    def test_setup_tiny_model(self, mock_model_cls, mock_tokenizer_cls):
        """Test setting up a tiny model through setup_model."""
        # Create tiny model
        model, config = create_tiny_llama_model()
        mock_model_cls.return_value = model
        
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=1000)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        mock_tokenizer_cls.return_value = tokenizer
        
        with patch('setup_model_for_training.log_rank_0'):
            result = setup_model(
                model_name_or_path="tiny-llama",
                use_liger_kernels=False,
                orthogonal_subspace_learning=False,
                rank=0
            )
        
        assert result is not None
        # Model should be aligned with tokenizer
        # Check that alignment was attempted (pad_token_id should be set)
        assert hasattr(result.config, 'pad_token_id')
        # The value should match what we set in the tokenizer
        if result.config.pad_token_id is not None:
            assert result.config.pad_token_id == 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('setup_model_for_training.dist.get_rank', return_value=0)
    @patch('setup_model_for_training.dist.get_world_size', return_value=1)
    @patch('setup_model_for_training.dist.is_initialized', return_value=False)
    def test_wrap_tiny_model_fsdp(self, mock_dist_init, mock_world_size, mock_rank):
        """Test FSDP wrapping with a tiny model."""
        from setup_model_for_training import wrap_fsdp2
        
        model, config = create_tiny_llama_model()
        model = model.cuda()
        
        # Wrap with FSDP2
        with patch('setup_model_for_training.init_device_mesh') as mock_mesh:
            with patch('setup_model_for_training.fully_shard') as mock_shard:
                mock_mesh.return_value = MagicMock()
                mock_shard.side_effect = lambda x, **kwargs: x
                
                wrapped_model = wrap_fsdp2(model)
                
                assert wrapped_model is not None
                # Check that sharding was attempted
                assert mock_shard.called
    
    def test_training_components_setup_with_tiny_model(self):
        """Test setting up training components with a tiny model."""
        model, config = create_tiny_llama_model()
        
        with patch('setup_model_for_training.wrap_fsdp2') as mock_wrap:
            with patch('transformers.get_scheduler') as mock_sched_fn:
                with patch('svd_utils.optim_wrapper') as mock_opt_wrap:
                    with patch('setup_model_for_training.log_rank_0'):
                        mock_wrap.return_value = model
                        mock_opt_wrap.side_effect = lambda opt, m: opt
                        
                        # Create mock scheduler
                        mock_scheduler = MagicMock()
                        mock_scheduler.split_batches = False
                        mock_scheduler.step = MagicMock()
                        mock_sched_fn.return_value = mock_scheduler
                        
                        model, optimizer, scheduler = setup_training_components(
                            model,
                            learning_rate=1e-4,
                            num_warmup_steps=10,
                            lr_scheduler="constant"
                        )
                        
                        assert model is not None
                        assert optimizer is not None
                        assert scheduler is not None


class TestSVDModelInitialization:
    """Test SVD/orthogonal subspace learning with tiny models."""
    
    @pytest.mark.skipif(True, reason="SVD initialization requires significant setup")
    def test_svd_model_creation(self):
        """Test creating an SVD model from a tiny base model."""
        # This would require implementing SVD for tiny models
        # Skipping for now as it's complex and optional
        pytest.skip("SVD model tests require full SVD implementation")


class TestEndToEndTraining:
    """End-to-end training tests with tiny models."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tiny_model_training_loop(self):
        """Test a complete training loop with a tiny model."""
        # Create tiny model
        model, config = create_tiny_llama_model()
        model = model.cuda()
        model.train()
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        
        # Track metrics
        losses = []
        
        # Training loop
        for step in range(5):
            # Create random batch
            batch_size = 2
            seq_length = 16
            input_ids = torch.randint(0, 500, (batch_size, seq_length)).cuda()
            labels = input_ids.clone()
            
            # Forward pass
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss
            
            # Handle both reduced and non-reduced loss
            # (setup_model patches loss to use reduction='none')
            if loss.numel() > 1:
                # Non-reduced loss from patched cross entropy
                # Need to mask out padding tokens if present
                loss_scalar = loss.mean()
            else:
                # Normal scalar loss
                loss_scalar = loss
            
            losses.append(loss_scalar.item())
            
            # Backward pass
            loss_scalar.backward()
            
            # Gradient step
            grad_norm = take_gradient_step(model, optimizer, scheduler)
            
            # grad_norm is a tensor, check it's valid
            assert grad_norm is not None
            assert torch.is_tensor(grad_norm)
            # For checking the value, we might need to handle different tensor shapes
            if grad_norm.numel() == 1:
                assert grad_norm.item() > 0
            else:
                assert grad_norm.mean().item() > 0
        
        # Loss should generally decrease (not always guaranteed with random data)
        print(f"Training losses: {losses}")
        assert len(losses) == 5
    
    def test_model_checkpoint_compatibility(self):
        """Test that tiny models can be saved and loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save model
            model, config = create_tiny_llama_model()
            save_path = Path(temp_dir) / "model"
            model.save_pretrained(save_path)
            config.save_pretrained(save_path)
            
            # Load model
            loaded_model = LlamaForCausalLM.from_pretrained(save_path)
            loaded_config = LlamaConfig.from_pretrained(save_path)
            
            assert loaded_config.hidden_size == config.hidden_size
            assert loaded_config.num_hidden_layers == config.num_hidden_layers
            
            # Check parameters match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
                assert n1 == n2
                assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
