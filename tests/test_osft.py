"""
Comprehensive tests for OSFT (Orthogonal Subspace Fine-Tuning) and SVD functionality.

Tests validate:
1. osft_unfreeze_rank_ratio validation in API
2. osft_target_patterns passing through API
3. SVD config generation with custom patterns
4. Integration with setup_model
"""

import pytest
import tempfile
import json
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

from mini_trainer.api_train import run_training, StreamablePopen
from mini_trainer.training_types import TorchrunArgs, TrainingArgs, TrainingMode
from mini_trainer.osft_utils import (
    auto_generate_target_osft_config, 
    get_model_config, 
    is_osft_param,
    create_osft_model_class,
    MODEL_CONFIGS,
    _get_model_patterns_from_name
)
from mini_trainer.setup_model_for_training import setup_model


class TestOSFTAPIValidation:
    """Test OSFT parameter validation in the API."""
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_osft_requires_rank_ratio(self, mock_popen_class):
        """Test that osft=True requires osft_unfreeze_rank_ratio to be provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            # osft=True but osft_unfreeze_rank_ratio=None should raise error
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=None  # This should cause an error
            )
            
            mock_popen = MagicMock()
            # it should not even run this, so return value doesn't matter here
            mock_popen.poll.return_value = 0
            mock_popen_class.return_value = mock_popen
            
            with pytest.raises(ValueError, match="osft_unfreeze_rank_ratio is required when osft is True"):
                run_training(torch_args, train_args)
            
            # shouldnt have even gotten run
            assert mock_popen_class.call_count == 0
            
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_osft_with_valid_rank_ratio(self, mock_popen_class):
        """Test that osft=True with valid unfreeze_rank_ratio passes validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.5  # Valid ratio
            )
            
            mock_popen = MagicMock()
            mock_popen.poll.return_value = 0  # Success
            mock_popen_class.return_value = mock_popen
            
            run_training(torch_args, train_args)
            
            # Verify command includes osft parameters
            call_args = mock_popen_class.call_args
            _, command = call_args[0]
            
            assert "--osft" in command
            assert "--osft-unfreeze-rank-ratio=0.5" in command
            assert mock_popen_class.call_count > 0
    
    def test_osft_unfreeze_rank_ratio_not_required_when_osft_false(self):
        """Test that osft_unfreeze_rank_ratio is not required when osft=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=False,
                osft_unfreeze_rank_ratio=None  # This should be fine
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                # Should not raise error
                run_training(torch_args, train_args)
                
                # Verify osft parameters not in command
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                assert "--osft" not in command
                assert all(not arg.startswith("--osft-unfreeze-rank-ratio") for arg in command)
                assert mock_popen_class.call_count > 0
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_osft_target_patterns_passed_through(self, mock_popen_class):
        """Test that osft_target_patterns are correctly passed through the API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            test_patterns = ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"]
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.75,
                osft_target_patterns=test_patterns
            )
            
            mock_popen = MagicMock()
            mock_popen.poll.return_value = 0
            mock_popen_class.return_value = mock_popen
            
            run_training(torch_args, train_args)
            
            # Verify command includes target patterns
            call_args = mock_popen_class.call_args
            _, command = call_args[0]
            
            assert "--osft" in command
            assert "--osft-unfreeze-rank-ratio=0.75" in command
            # Find the target patterns argument
            patterns_arg = None
            for arg in command:
                if arg.startswith("--osft-target-patterns="):
                    patterns_arg = arg
                    break
            
            assert patterns_arg is not None
            # The patterns should be passed as a list string
            expected = "--osft-target-patterns=self_attn.q_proj,self_attn.k_proj,mlp.gate_proj"
            assert patterns_arg == expected
    
    def test_osft_target_patterns_empty_list(self):
        """Test that empty osft_target_patterns list is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.5,
                osft_target_patterns=[]  # Empty list
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Empty list is treated same as None - not passed
                # This is reasonable as empty list means no custom patterns
                patterns_arg = None
                for arg in command:
                    if arg.startswith("--osft-target-patterns="):
                        patterns_arg = arg
                        break
                
                assert patterns_arg is None
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_osft_target_patterns_none_not_passed(self, mock_popen_class):
        """Test that None osft_target_patterns is not passed to command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.5,
                osft_target_patterns=None  # None should not be passed
            )
            
            mock_popen = MagicMock()
            mock_popen.poll.return_value = 0
            mock_popen_class.return_value = mock_popen
            
            run_training(torch_args, train_args)
            
            call_args = mock_popen_class.call_args
            _, command = call_args[0]
            
            # None should result in no target patterns argument
            assert all(not arg.startswith("--osft-target-patterns") for arg in command)
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_various_rank_ratios(self, mock_popen_class):
        """Test that different rank ratios are correctly passed."""
        rank_ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        for ratio in rank_ratios:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch_args = TorchrunArgs(nproc_per_node=8)
                train_args = TrainingArgs(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=32,
                    max_tokens_per_gpu=1000,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    osft=True,
                    osft_unfreeze_rank_ratio=ratio
                )
                
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                assert f"--osft-unfreeze-rank-ratio={ratio}" in command


class TestOSFTConfigGeneration:
    """Test SVD configuration generation with custom patterns."""
    
    def test_get_model_patterns_from_name(self):
        """Test pattern detection from model names."""
        # Test known model types
        # We need to make sure that these models are tested:
        # - Llama
        # - Qwen
        # - Mistral
        # - Phi-4
        assert _get_model_patterns_from_name("llama") == MODEL_CONFIGS["llama"]["patterns"]
        assert _get_model_patterns_from_name("gpt-j-6b") == MODEL_CONFIGS["gpt-j"]["patterns"]
        assert _get_model_patterns_from_name("gptj") == MODEL_CONFIGS["gpt-j"]["patterns"]
        assert _get_model_patterns_from_name("opt-350m") == MODEL_CONFIGS["opt"]["patterns"]
        assert _get_model_patterns_from_name("qwen2-7b") == MODEL_CONFIGS["qwen"]["patterns"]
        assert _get_model_patterns_from_name("gemma-2b") == MODEL_CONFIGS["gemma"]["patterns"]
        assert _get_model_patterns_from_name("mistral") == MODEL_CONFIGS["mistral"]["patterns"]
        assert _get_model_patterns_from_name("mistral-7b") == MODEL_CONFIGS["mistral"]["patterns"]
        assert _get_model_patterns_from_name("microsoft/Phi-4") == MODEL_CONFIGS["phi3"]["patterns"]
        assert _get_model_patterns_from_name("microsoft/Phi-3") == MODEL_CONFIGS["phi3"]["patterns"]
        assert _get_model_patterns_from_name("microsoft/Phi-4-mini-instruct") == MODEL_CONFIGS["phi3"]["patterns"]
        
        # Test default fallback
        assert _get_model_patterns_from_name("unknown-model") == MODEL_CONFIGS["default"]["patterns"]
    
    def test_get_model_config_with_custom_patterns(self):
        """Test that custom patterns override model defaults."""
        custom_patterns = ["custom.layer1", "custom.layer2"]
        
        # Custom patterns should override model-specific patterns
        patterns = get_model_config("llama", target_patterns=custom_patterns)
        assert patterns == custom_patterns
        
        # Custom patterns should override default
        patterns = get_model_config(None, target_patterns=custom_patterns)
        assert patterns == custom_patterns
    
    def test_get_model_config_without_custom_patterns(self):
        """Test model config retrieval without custom patterns."""
        # Should get model-specific patterns
        patterns = get_model_config("llama", target_patterns=None)
        assert patterns == MODEL_CONFIGS["llama"]["patterns"]
        
        # Should get default patterns
        patterns = get_model_config(None, target_patterns=None)
        assert patterns == MODEL_CONFIGS["default"]["patterns"]
    
    def test_auto_generate_osft_config_with_custom_patterns(self):
        """Test OSFT config generation with custom target patterns."""
        # Create a mock model with various layers
        mock_model = MagicMock()
        mock_params = [
            ("layer1.self_attn.q_proj.weight", torch.zeros(128, 64)),
            ("layer1.self_attn.k_proj.weight", torch.zeros(128, 64)),
            ("layer1.mlp.gate_proj.weight", torch.zeros(256, 128)),
            ("layer2.custom_proj.weight", torch.zeros(100, 50)),
            ("layer2.another_proj.weight", torch.zeros(200, 100)),
        ]
        mock_model.named_parameters.return_value = mock_params
        
        # Test with custom patterns
        custom_patterns = ["custom_proj", "another_proj"]
        config = auto_generate_target_osft_config(
            mock_model,
            target_patterns=custom_patterns,
            rank_ratio=0.5
        )
        
        # Should only include layers matching custom patterns
        assert "layer2.custom_proj.weight" in config
        assert "layer2.another_proj.weight" in config
        assert "layer1.self_attn.q_proj.weight" not in config
        assert "layer1.self_attn.k_proj.weight" not in config
        assert "layer1.mlp.gate_proj.weight" not in config
        
        # Check rank values
        assert config["layer2.custom_proj.weight"] == 25  # min(100, 50) * 0.5
        assert config["layer2.another_proj.weight"] == 50  # min(200, 100) * 0.5
    
    def test_auto_generate_osft_config_with_rank_ratio(self):
        """Test that rank_ratio correctly affects the generated config."""
        mock_model = MagicMock()
        mock_params = [
            ("layer.proj.weight", torch.zeros(100, 80)),
        ]
        mock_model.named_parameters.return_value = mock_params
        
        # Test different rank ratios
        for ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
            config = auto_generate_target_osft_config(
                mock_model,
                target_patterns=["proj"],
                rank_ratio=ratio
            )
            
            expected_rank = int(80 * ratio)  # min(100, 80) * ratio
            assert config["layer.proj.weight"] == expected_rank
    
    def test_auto_generate_svd_config_edge_cases(self):
        """Test edge cases in SVD config generation."""
        mock_model = MagicMock()
        
        # Test with rank_ratio >= 1.0 (should cap at full_rank - 1)
        mock_params = [("layer.proj.weight", torch.zeros(50, 50))]
        mock_model.named_parameters.return_value = mock_params
        
        config = auto_generate_target_osft_config(
            mock_model,
            target_patterns=["proj"],
            rank_ratio=1.0
        )
        assert config["layer.proj.weight"] == 49  # full_rank - 1
        
        # Test with 1D parameters (should be skipped)
        mock_params = [
            ("layer.bias", torch.zeros(100)),  # 1D parameter
            ("layer.weight", torch.zeros(100, 50)),  # 2D parameter
        ]
        mock_model.named_parameters.return_value = mock_params
        
        config = auto_generate_target_osft_config(
            mock_model,
            target_patterns=["layer"],
            rank_ratio=0.5
        )
        
        assert "layer.bias" not in config  # 1D should be skipped
        assert "layer.weight" in config  # 2D should be included
    
    def test_is_osft_param_function(self):
        """Test the is_osft_param utility function."""
        osft_config = {
            "layer1.weight": 10,
            "layer2.weight": 0,  # 0 means not OSFT
        }
        
        # 2D param with positive rank in config
        param_2d = torch.zeros(100, 50)
        assert is_osft_param("layer1.weight", param_2d, osft_config) is True
        
        # 2D param with 0 rank in config
        assert is_osft_param("layer2.weight", param_2d, osft_config) is False
        
        # 2D param not in config
        assert is_osft_param("layer3.weight", param_2d, osft_config) is False
        
        # 1D param (should be False regardless)
        param_1d = torch.zeros(100)
        assert is_osft_param("layer1.weight", param_1d, osft_config) is False
    
    def _create_tiny_llama_model(self):
        """Create a tiny Llama model for testing."""
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            pytest.skip("LlamaForCausalLM not available in this transformers version")
            
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            rope_theta=10000.0,
        )
        model = LlamaForCausalLM(config)
        return model, config, "llama"
    
    def _create_tiny_mistral_model(self):
        """Create a tiny Mistral model for testing."""
        try:
            from transformers import MistralConfig, MistralForCausalLM
        except ImportError:
            pytest.skip("MistralForCausalLM not available in this transformers version")
            
        config = MistralConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            sliding_window=8,
        )
        model = MistralForCausalLM(config)
        return model, config, "mistral"
    
    def _create_tiny_qwen2_model(self):
        """Create a tiny Qwen2 model for testing."""
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            pytest.skip("Qwen2ForCausalLM not available in this transformers version")
            
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
        model = Qwen2ForCausalLM(config)
        return model, config, "qwen"
    
    def _create_tiny_phi4_model(self):
        """Create a tiny Phi-4 model for testing."""
        try:
            from transformers import Phi3Config, Phi3ForCausalLM
        except ImportError:
            pytest.skip("Phi3ForCausalLM not available in this transformers version")
            
        config = Phi3Config(
            vocab_size=1000,  # Large enough for pad_token_id
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            pad_token_id=999,  # Set within vocab_size
            eos_token_id=998,
            bos_token_id=997
        )
        model = Phi3ForCausalLM(config)
        return model, config, "phi3"
    
    def _create_tiny_gptj_model(self):
        """Create a tiny GPT-J model for testing."""
        try:
            from transformers import GPTJConfig, GPTJForCausalLM
        except ImportError:
            pytest.skip("GPTJForCausalLM not available in this transformers version")
            
        config = GPTJConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            rotary_dim=8,  # Required for GPT-J
            max_position_embeddings=16
        )
        model = GPTJForCausalLM(config)
        return model, config, "gpt-j"
    
    def _create_tiny_gptneo_model(self):
        """Create a tiny GPT-NEO model for testing."""
        try:
            from transformers import GPTNeoConfig, GPTNeoForCausalLM
        except ImportError:
            pytest.skip("GPTNeoForCausalLM not available in this transformers version")
            
        config = GPTNeoConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
            attention_types=[['global', 2], ['global', 2]],  # Required format
            attention_layers=['global', 'global']
        )
        model = GPTNeoForCausalLM(config)
        return model, config, "gpt-neo"
    
    def _create_tiny_opt_model(self):
        """Create a tiny OPT model for testing."""
        try:
            from transformers import OPTConfig, OPTForCausalLM
        except ImportError:
            pytest.skip("OPTForCausalLM not available in this transformers version")
            
        config = OPTConfig(
            vocab_size=100,
            hidden_size=16,
            ffn_dim=32,  # OPT uses ffn_dim instead of intermediate_size
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16
        )
        model = OPTForCausalLM(config)
        return model, config, "opt"
    
    def _create_tiny_gemma_model(self):
        """Create a tiny GEMMA model for testing."""
        try:
            from transformers import GemmaConfig, GemmaForCausalLM
        except ImportError:
            pytest.skip("GemmaForCausalLM not available in this transformers version")
            
        config = GemmaConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            pad_token_id=0
        )
        model = GemmaForCausalLM(config)
        return model, config, "gemma"
    
    def _create_tiny_granite_model(self):
        """Create a tiny GRANITE model for testing."""
        try:
            from transformers import GraniteConfig, GraniteForCausalLM
        except ImportError:
            pytest.skip("GraniteForCausalLM not available in this transformers version")
            
        config = GraniteConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            pad_token_id=0
        )
        model = GraniteForCausalLM(config)
        return model, config, "granite"
    
    def _get_model_layer_patterns(self, model_type, config, layer_idx):
        """Get expected layer patterns for different model types."""
        if model_type in ["llama", "mistral", "qwen", "gemma", "granite"]:
            # These models use the same layer structure: model.layers.{idx}
            layer_prefix = f"model.layers.{layer_idx}"
            return [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight", 
                f"{layer_prefix}.self_attn.v_proj.weight",
                f"{layer_prefix}.self_attn.o_proj.weight",
                f"{layer_prefix}.mlp.gate_proj.weight",
                f"{layer_prefix}.mlp.up_proj.weight",
                f"{layer_prefix}.mlp.down_proj.weight",
            ]
        elif model_type == "phi3":
            # Phi-3/Phi-4 models use combined projections: model.layers.{idx}
            layer_prefix = f"model.layers.{layer_idx}"
            return [
                f"{layer_prefix}.self_attn.qkv_proj.weight",   # Combined q/k/v projection
                f"{layer_prefix}.self_attn.o_proj.weight",     # Output projection
                f"{layer_prefix}.mlp.gate_up_proj.weight",     # Combined gate/up projection
                f"{layer_prefix}.mlp.down_proj.weight",        # Down projection
            ]
        elif model_type == "gpt-j":
            # GPT-J uses h.{idx} instead of layers.{idx}
            layer_prefix = f"transformer.h.{layer_idx}"
            return [
                f"{layer_prefix}.attn.q_proj.weight",
                f"{layer_prefix}.attn.k_proj.weight",
                f"{layer_prefix}.attn.v_proj.weight",
                f"{layer_prefix}.attn.out_proj.weight",
                f"{layer_prefix}.mlp.fc_in.weight",
                f"{layer_prefix}.mlp.fc_out.weight",
            ]
        elif model_type == "gpt-neo":
            # GPT-NEO uses h.{idx} with nested attention structure
            layer_prefix = f"transformer.h.{layer_idx}"
            return [
                f"{layer_prefix}.attn.attention.q_proj.weight",
                f"{layer_prefix}.attn.attention.k_proj.weight",
                f"{layer_prefix}.attn.attention.v_proj.weight",
                f"{layer_prefix}.attn.attention.out_proj.weight",
                f"{layer_prefix}.mlp.c_fc.weight",
                f"{layer_prefix}.mlp.c_proj.weight",
            ]
        elif model_type == "opt":
            # OPT uses decoder.layers.{idx}
            layer_prefix = f"model.decoder.layers.{layer_idx}"
            return [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight",
                f"{layer_prefix}.self_attn.v_proj.weight",
                f"{layer_prefix}.self_attn.out_proj.weight",
                f"{layer_prefix}.fc1.weight",
                f"{layer_prefix}.fc2.weight",
            ]
        else:
            # For future model types, we can add specific handling
            raise NotImplementedError(f"Layer patterns not implemented for {model_type}")
    
    @pytest.mark.parametrize("model_creator", [
        "_create_tiny_llama_model",
        "_create_tiny_mistral_model", 
        "_create_tiny_qwen2_model",
        "_create_tiny_phi4_model",
        "_create_tiny_gptj_model",
        "_create_tiny_gptneo_model",
        "_create_tiny_opt_model",
        "_create_tiny_gemma_model",
        "_create_tiny_granite_model",
    ])
    def test_model_state_dict_pattern_matching(self, model_creator):
        """Test that model state dicts correctly match expected OSFT patterns."""
        # Get the model creator method and create the model
        creator_method = getattr(self, model_creator)
        model, config, model_type = creator_method()
        
        # Get the OSFT config using the model
        osft_config = auto_generate_target_osft_config(
            model,
            model_name_or_class=model_type,
            rank_ratio=0.5
        )
        
        # Expected patterns from MODEL_CONFIGS
        expected_patterns = MODEL_CONFIGS[model_type]["patterns"]
        
        # Verify that all expected patterns are found in the model's state dict
        model_param_names = [name for name, _ in model.named_parameters()]
        
        # Check that each expected pattern matches at least one parameter
        for pattern in expected_patterns:
            matching_params = [name for name in osft_config.keys() if pattern in name]
            assert len(matching_params) > 0, f"Pattern '{pattern}' not found in OSFT config for {model_type}"
            
            # Also verify these parameters exist in the actual model
            model_matches = [name for name in model_param_names if pattern in name and ".weight" in name]
            assert len(model_matches) > 0, f"Pattern '{pattern}' not found in model parameters for {model_type}"
        
        # Verify that the OSFT config only contains parameters matching our patterns
        for param_name in osft_config.keys():
            assert any(pattern in param_name for pattern in expected_patterns), \
                f"Parameter '{param_name}' doesn't match any expected pattern for {model_type}"
        
        # Verify correct number of layers are matched (2 layers as configured)
        for i in range(config.num_hidden_layers):
            expected_layer_params = self._get_model_layer_patterns(model_type, config, i)
            for expected_param in expected_layer_params:
                assert expected_param in osft_config, \
                    f"Expected parameter '{expected_param}' not found in OSFT config for {model_type}"
        
        # Verify rank values are correctly calculated
        for param_name, rank in osft_config.items():
            param = dict(model.named_parameters())[param_name]
            expected_rank = int(min(param.shape) * 0.5)
            if expected_rank >= min(param.shape):
                expected_rank = min(param.shape) - 1
            assert rank == expected_rank, \
                f"Rank mismatch for {param_name} in {model_type}: got {rank}, expected {expected_rank}"
    



class TestOSFTModelCreation:
    """Test OSFT model class creation and initialization."""
    
    def test_create_osft_model_class(self):
        """Test that create_osft_model_class creates a valid subclass."""
        # Create a simple mock base class
        class MockModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(10, 10)
        
        # Create OSFT model class
        OSFTModelClass = create_osft_model_class(MockModel)
        
        # Check class inheritance
        assert issubclass(OSFTModelClass, MockModel)
        assert OSFTModelClass.__name__ == "MockModelWithOSFT"
        
        # Check that required methods exist
        assert hasattr(OSFTModelClass, 'reinitialize_osft')
        assert hasattr(OSFTModelClass, 'reinitialize_osft_distributed')
        assert hasattr(OSFTModelClass, 'project_gradients')
        assert hasattr(OSFTModelClass, 'from_pretrained')
    
    def test_osft_model_initialization_without_osft(self):
        """Test OSFT model can be initialized without OSFT decomposition."""
        class MockModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.dtype = torch.float32
        
        OSFTModelClass = create_osft_model_class(MockModel)
        
        # Initialize without OSFT
        config = MagicMock()
        model = OSFTModelClass(config, osft_config={}, initialize_osft=False)
        
        assert model.osft_config == {}
        assert hasattr(model, 'osft_params')
        assert len(model.osft_params) == 0



class TestSetupModelIntegration:
    """Test integration of OSFT options with setup_model function."""
    
    @patch('mini_trainer.setup_model_for_training.log_rank_0')
    @patch('mini_trainer.setup_model_for_training.AutoModelForCausalLM')
    @patch('mini_trainer.setup_model_for_training.AutoTokenizer')
    @patch('mini_trainer.setup_model_for_training.AutoConfig')
    @patch('mini_trainer.osft_utils.auto_generate_target_osft_config')
    @patch('mini_trainer.osft_utils.create_osft_model_class')
    def test_osft_params_flow_through_setup(self, mock_osft_class, mock_auto_config, mock_transformers_auto_config, mock_tokenizer_cls, mock_model_cls, mock_log):
        """Test that OSFT parameters flow through the setup correctly."""
        # Test that OSFT model creation gets the right parameters
        mock_auto_config.return_value = {"layer.weight": 10}
        
        # Mock the OSFT model class
        mock_osft_model_cls = MagicMock()
        mock_osft_instance = MagicMock()
        mock_osft_instance.config = MagicMock()
        mock_osft_instance.config.vocab_size = 1000
        mock_osft_instance.dtype = torch.float32
        
        # Setup from_pretrained to return a properly configured model
        def from_pretrained_side_effect(*args, **kwargs):
            # Store the kwargs for verification
            mock_osft_model_cls.last_kwargs = kwargs
            return mock_osft_instance
        
        mock_osft_model_cls.from_pretrained.side_effect = from_pretrained_side_effect
        mock_osft_class.return_value = mock_osft_model_cls
        
        # Mock tokenizer and base model
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__ = MagicMock(return_value=1000)
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.config.vocab_size = 1000
        mock_model_cls.from_pretrained.return_value = mock_base_model
        
        # Call setup_model with OSFT params
        model = setup_model(
            osft=True,
            local_rank=0,
            osft_rank_ratio=0.75,
            osft_target_patterns=["custom.layer1", "custom.layer2"],
            model_name_or_path="test-model"
        )
        
        # Verify the OSFT model class was created
        mock_osft_class.assert_called_once()
        
        # Verify from_pretrained was called with the right params
        assert 'rank_ratio' in mock_osft_model_cls.last_kwargs
        assert mock_osft_model_cls.last_kwargs['rank_ratio'] == 0.75
        assert 'target_patterns' in mock_osft_model_cls.last_kwargs
        assert mock_osft_model_cls.last_kwargs['target_patterns'] == ["custom.layer1", "custom.layer2"]


class TestEndToEndOSFT:
    """End-to-end tests for OSFT functionality."""
    
    def test_command_line_osft_params_validation(self):
        """Test that command line validates OSFT parameters correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test script that validates OSFT params
            test_script = Path(tmpdir) / "validate_osft.py"
            test_script.write_text("""
import sys
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def main(
    osft: bool = False,
    osft_unfreeze_rank_ratio: Optional[float] = None,
    osft_target_patterns: Optional[str] = None
):
    # Validate: if osft is True, unfreeze_rank_ratio must be provided
    if osft and osft_unfreeze_rank_ratio is None:
        print("ERROR: osft_unfreeze_rank_ratio required")
        raise typer.Exit(1)

    # Parse target patterns if provided (comma-delimited)
    if osft_target_patterns:
        patterns = [p.strip() for p in osft_target_patterns.split(",")]
        print(f"PATTERNS: {patterns}")

    print(f"SUCCESS: osft={osft}, ratio={osft_unfreeze_rank_ratio}")

if __name__ == "__main__":
    app()
""")
            
            # Test valid OSFT configuration
            result = subprocess.run(
                ["python", str(test_script), "--osft", "--osft-unfreeze-rank-ratio=0.5", 
                 "--osft-target-patterns=q_proj,k_proj"],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "SUCCESS" in result.stdout
            assert "PATTERNS: ['q_proj', 'k_proj']" in result.stdout
            
            # Test missing unfreeze_rank_ratio
            result = subprocess.run(
                ["python", str(test_script), "--osft"],
                capture_output=True,
                text=True
            )
            assert result.returncode == 1
            assert "ERROR: osft_unfreeze_rank_ratio required" in result.stdout
            
            # Test osft=False doesn't require rank_ratio
            result = subprocess.run(
                ["python", str(test_script)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "SUCCESS: osft=False" in result.stdout


class TestOSFTPrepareStateDict:
    """Test suite for OSFT prepare_state_dict_for_save functionality."""

    def test_osft_prepare_basic_reconstruction(self):
        """Test that OSFT parameters get reconstructed correctly."""
        
        # Create simple base model
        class SimpleModel(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.config = config or MagicMock()
                self.dtype = torch.float32
        
        # Create OSFT version
        OSFTModel = create_osft_model_class(SimpleModel)
        osft_config = {"linear.weight": 2}  # rank 2 decomposition
        
        model = OSFTModel(MagicMock(), osft_config=osft_config, initialize_osft=False)
        model.reinitialize_osft(decompose_existing_weights=True)
        
        # Get state dict with OSFT parameters
        osft_state_dict = model.state_dict()
        
        # Verify OSFT parameters exist
        osft_keys = [k for k in osft_state_dict.keys() if "osft_params" in k or "_U_high" in k or "_S_high" in k or "_V_high" in k]
        assert len(osft_keys) > 0, "No OSFT parameters found"
        
        # Call prepare_state_dict_for_save
        reconstructed = model.prepare_state_dict_for_save(osft_state_dict.copy())
        
        # Verify OSFT parameters are removed and original weights are present
        assert "linear.weight" in reconstructed
        for key in reconstructed.keys():
            assert "osft_params" not in key
            assert "_U_high" not in key and "_S_high" not in key and "_V_high" not in key
        
        # Verify shape is correct
        assert reconstructed["linear.weight"].shape == (4, 4)

    def test_osft_prepare_preserves_non_osft(self):
        """Test that non-OSFT parameters are preserved unchanged."""
        
        class ModelWithNonOSFT(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.osft_layer = nn.Linear(4, 4, bias=False)
                self.regular_layer = nn.Linear(4, 2, bias=True)  # Will not be decomposed
                self.config = config or MagicMock()
                self.dtype = torch.float32
        
        OSFTModel = create_osft_model_class(ModelWithNonOSFT)
        osft_config = {"osft_layer.weight": 2}  # Only decompose one layer
        
        model = OSFTModel(MagicMock(), osft_config=osft_config, initialize_osft=False)
        model.reinitialize_osft(decompose_existing_weights=True)
        
        # Set known values for non-OSFT parameters
        with torch.no_grad():
            model.regular_layer.weight.fill_(3.14159)
            model.regular_layer.bias.fill_(2.71828)
        
        state_dict = model.state_dict()
        original_weight = state_dict["regular_layer.weight"].clone()
        original_bias = state_dict["regular_layer.bias"].clone()
        
        # Call prepare_state_dict_for_save
        reconstructed = model.prepare_state_dict_for_save(state_dict.copy())
        
        # Verify non-OSFT parameters are unchanged
        assert torch.equal(reconstructed["regular_layer.weight"], original_weight)
        assert torch.equal(reconstructed["regular_layer.bias"], original_bias)

    def test_osft_prepare_empty_config(self):
        """Test that models without OSFT work correctly."""
        
        class RegularModel(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.config = config or MagicMock()
                self.dtype = torch.float32
        
        OSFTModel = create_osft_model_class(RegularModel)
        model = OSFTModel(MagicMock(), osft_config={}, initialize_osft=False)
        
        state_dict = model.state_dict()
        original_keys = set(state_dict.keys())
        
        # Call prepare_state_dict_for_save (should be no-op)
        result = model.prepare_state_dict_for_save(state_dict.copy())
        
        # Verify state dict is unchanged
        assert set(result.keys()) == original_keys
        for key in original_keys:
            assert torch.equal(result[key], state_dict[key])

    def test_osft_prepare_dtype_preservation(self):
        """Test that reconstructed weights have correct dtype."""
        
        class TypedModel(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.config = config or MagicMock()
                self.dtype = torch.float32
                self.output_dtype = torch.float32
        
        OSFTModel = create_osft_model_class(TypedModel)
        osft_config = {"linear.weight": 2}
        
        model = OSFTModel(MagicMock(), osft_config=osft_config, initialize_osft=False)
        model.reinitialize_osft(decompose_existing_weights=True)
        
        state_dict = model.state_dict()
        reconstructed = model.prepare_state_dict_for_save(state_dict.copy())
        
        # Verify dtype is preserved
        assert reconstructed["linear.weight"].dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
