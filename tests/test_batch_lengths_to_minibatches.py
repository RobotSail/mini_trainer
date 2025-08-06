"""
Test suite for batch_lengths_to_minibatches function.

Tests both correctness and efficiency of the batching algorithm,
measuring load distribution and number of minibatches generated.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import time
from sampler import batch_lengths_to_minibatches
from batch_packer import batch_lengths_to_minibatches_lpt


class TestBatchLengthsToMinibatches:
    def test_empty_batch(self):
        """Test with empty batch."""
        result = batch_lengths_to_minibatches([], 130000, 4, 0)
        assert result == []

    def test_single_sequence(self):
        """Test with single sequence."""
        result = batch_lengths_to_minibatches([5000], 130000, 4, 0)
        assert len(result) == 1
        assert result[0] == [0]

        # Other ranks should get padding
        result_rank1 = batch_lengths_to_minibatches([5000], 130000, 4, 1)
        assert result_rank1 == [[-1]]

    def test_basic_distribution(self):
        """Test basic sequence distribution across ranks."""
        batch_lengths = [10000, 20000, 30000, 40000]
        max_tokens = 50000
        num_ranks = 2

        # Get results for both ranks
        rank0_result = batch_lengths_to_minibatches(
            batch_lengths, max_tokens, num_ranks, 0
        )
        rank1_result = batch_lengths_to_minibatches(
            batch_lengths, max_tokens, num_ranks, 1
        )

        # Should have same number of minibatches
        assert len(rank0_result) == len(rank1_result)

        # Check no sequence exceeds max_tokens per rank
        for minibatch in rank0_result:
            total_tokens = sum(batch_lengths[i] for i in minibatch if i != -1)
            assert total_tokens <= max_tokens

        for minibatch in rank1_result:
            total_tokens = sum(batch_lengths[i] for i in minibatch if i != -1)
            assert total_tokens <= max_tokens

    def test_no_duplicates_across_ranks(self):
        """Test that no sequence appears in multiple ranks."""
        batch_lengths = [10000, 20000, 30000, 40000, 50000]
        max_tokens = 60000
        num_ranks = 3

        all_assigned_indices = set()
        for rank in range(num_ranks):
            rank_result = batch_lengths_to_minibatches(
                batch_lengths, max_tokens, num_ranks, rank
            )
            for minibatch in rank_result:
                for idx in minibatch:
                    if idx != -1:
                        assert idx not in all_assigned_indices, (
                            f"Index {idx} assigned to multiple ranks"
                        )
                        all_assigned_indices.add(idx)

        # All non-padding indices should be covered
        expected_indices = set(range(len(batch_lengths)))
        assert all_assigned_indices == expected_indices

    def test_token_limit_enforcement(self):
        """Test that token limits are strictly enforced."""
        batch_lengths = [80000, 40000, 30000, 20000]
        max_tokens = 100000
        num_ranks = 2

        for rank in range(num_ranks):
            rank_result = batch_lengths_to_minibatches(
                batch_lengths, max_tokens, num_ranks, rank
            )
            for minibatch in rank_result:
                total_tokens = sum(batch_lengths[i] for i in minibatch if i != -1)
                assert total_tokens <= max_tokens, (
                    f"Rank {rank} exceeded token limit: {total_tokens} > {max_tokens}"
                )


class TestBatchingEfficiency:
    """Test suite for measuring batching algorithm efficiency."""

    def generate_realistic_lengths(self, n_sequences=1000, seed=42):
        """Generate realistic sequence lengths from 100 to 100k tokens."""
        np.random.seed(seed)

        # Realistic distribution: mostly shorter sequences with some long ones
        short_seqs = np.random.randint(
            100, 2000, size=int(n_sequences * 0.5)
        )  # 50% short
        medium_seqs = np.random.randint(
            2000, 20000, size=int(n_sequences * 0.3)
        )  # 30% medium
        long_seqs = np.random.randint(
            20000, 60000, size=int(n_sequences * 0.15)
        )  # 15% long
        very_long_seqs = np.random.randint(
            60000, 100000, size=int(n_sequences * 0.05)
        )  # 5% very long

        lengths = np.concatenate([short_seqs, medium_seqs, long_seqs, very_long_seqs])
        np.random.shuffle(lengths)
        return lengths.tolist()

    def measure_load_distribution(self, batch_lengths, max_tokens, num_ranks=8):
        """Measure load distribution with focus on per-minibatch (row) balance."""
        # Collect results for all ranks (columns)
        all_rank_results = []
        for rank in range(num_ranks):
            rank_result = batch_lengths_to_minibatches(
                batch_lengths, max_tokens, num_ranks, rank
            )
            all_rank_results.append(rank_result)

        # Determine total minibatches (rows in the matrix)
        total_minibatches = max(len(r) for r in all_rank_results)

        # Analyze each minibatch (row) separately
        minibatch_balance_scores = []
        minibatch_efficiencies = []
        total_tokens_used = 0

        for mb_idx in range(total_minibatches):
            # Collect loads for this row across all columns (ranks)
            row_loads = []
            row_lengths = []  # Individual sequence lengths in this row

            for rank in range(num_ranks):
                if mb_idx < len(all_rank_results[rank]):
                    rank_mb = all_rank_results[rank][mb_idx]
                    # Get actual sequence lengths (not indices)
                    seq_lengths = [batch_lengths[i] for i in rank_mb if i != -1]
                    row_lengths.extend(seq_lengths)
                    row_loads.append(sum(seq_lengths))
                else:
                    row_loads.append(0)

            # Calculate balance metrics for this row
            if row_lengths:  # If there are any sequences in this row
                # Variance of loads across ranks in this row
                load_variance = np.var(row_loads)
                load_std = np.std(row_loads)

                # Balance ratio: min/max load (closer to 1 is better)
                non_zero_loads = [l for l in row_loads if l > 0]
                if non_zero_loads:
                    balance_ratio = min(non_zero_loads) / max(non_zero_loads)
                else:
                    balance_ratio = 1.0

                # Efficiency: how well we utilize capacity in this row
                row_efficiency = sum(row_loads) / (num_ranks * max_tokens)

                minibatch_balance_scores.append(balance_ratio)
                minibatch_efficiencies.append(row_efficiency)
                total_tokens_used += sum(row_loads)

        # Calculate aggregate statistics
        avg_row_balance = (
            np.mean(minibatch_balance_scores) if minibatch_balance_scores else 0
        )
        worst_row_balance = (
            np.min(minibatch_balance_scores) if minibatch_balance_scores else 0
        )
        avg_row_efficiency = (
            np.mean(minibatch_efficiencies) if minibatch_efficiencies else 0
        )

        # Overall efficiency considering wasted capacity
        overall_efficiency = total_tokens_used / (
            total_minibatches * num_ranks * max_tokens
        )

        return {
            "total_minibatches": total_minibatches,
            "avg_row_balance_ratio": avg_row_balance,
            "worst_row_balance_ratio": worst_row_balance,
            "row_balance_std": np.std(minibatch_balance_scores)
            if minibatch_balance_scores
            else 0,
            "avg_row_efficiency": avg_row_efficiency,
            "overall_efficiency": overall_efficiency,
            "total_sequences": len(batch_lengths),
            "total_tokens": sum(batch_lengths),
            # For backward compatibility
            "load_balance_ratio": avg_row_balance,
            "efficiency": overall_efficiency,
        }

    def test_small_batch_efficiency(self):
        """Test efficiency on small batches."""
        batch_lengths = [10000, 25000, 35000, 50000, 60000, 75000, 80000, 90000]
        max_tokens = 130000
        num_ranks = 4

        stats = self.measure_load_distribution(batch_lengths, max_tokens, num_ranks)

        print(f"\nSmall batch efficiency stats:")
        print(f"  Total minibatches: {stats['total_minibatches']}")
        print(f"  Avg row balance ratio: {stats['avg_row_balance_ratio']:.3f}")
        print(f"  Worst row balance ratio: {stats['worst_row_balance_ratio']:.3f}")
        print(f"  Overall efficiency: {stats['overall_efficiency']:.3f}")

        # Basic sanity checks
        assert stats["total_minibatches"] > 0
        assert stats["load_balance_ratio"] > 0
        assert stats["efficiency"] > 0

    def test_large_batch_efficiency(self):
        """Test efficiency on large realistic batches."""
        batch_lengths = self.generate_realistic_lengths(n_sequences=2000)
        max_tokens = 130000
        num_ranks = 8

        stats = self.measure_load_distribution(batch_lengths, max_tokens, num_ranks)

        print(f"\nLarge batch efficiency stats:")
        print(f"  Total sequences: {len(batch_lengths)}")
        print(f"  Total minibatches: {stats['total_minibatches']}")
        print(f"  Avg row balance ratio: {stats['avg_row_balance_ratio']:.3f}")
        print(f"  Worst row balance ratio: {stats['worst_row_balance_ratio']:.3f}")
        print(f"  Overall efficiency: {stats['overall_efficiency']:.3f}")

        # Store baseline for comparison
        return stats

    def test_varying_sequence_lengths(self):
        """Test with highly variable sequence lengths (realistic range)."""
        # Create batch with extreme length variations in realistic range
        short_seqs = [np.random.randint(100, 1000) for _ in range(100)]
        medium_seqs = [np.random.randint(5000, 15000) for _ in range(50)]
        long_seqs = [np.random.randint(40000, 80000) for _ in range(20)]
        very_long_seqs = [np.random.randint(80000, 100000) for _ in range(5)]

        batch_lengths = short_seqs + medium_seqs + long_seqs + very_long_seqs
        np.random.shuffle(batch_lengths)

        max_tokens = 130000
        num_ranks = 8

        stats = self.measure_load_distribution(batch_lengths, max_tokens, num_ranks)

        print(f"\nVariable lengths efficiency stats:")
        print(f"  Total sequences: {len(batch_lengths)}")
        print(f"  Total minibatches: {stats['total_minibatches']}")
        print(f"  Avg row balance ratio: {stats['avg_row_balance_ratio']:.3f}")
        print(f"  Worst row balance ratio: {stats['worst_row_balance_ratio']:.3f}")
        print(f"  Overall efficiency: {stats['overall_efficiency']:.3f}")

        # With current greedy algorithm, load balance might be poor
        # This test establishes baseline for improvement
        return stats

    def test_edge_cases(self):
        """Test edge cases that might break the algorithm."""
        # Case 1: All sequences same length
        uniform_lengths = [15000] * 20
        stats1 = self.measure_load_distribution(uniform_lengths, 130000, 4)
        print(
            f"\nUniform lengths - Avg row balance: {stats1['avg_row_balance_ratio']:.3f}"
        )

        # Case 2: One very long sequence with many short ones
        mixed_lengths = [1000] * 50 + [95000]
        stats2 = self.measure_load_distribution(mixed_lengths, 130000, 4)
        print(
            f"Mixed with outlier - Avg row balance: {stats2['avg_row_balance_ratio']:.3f}"
        )

        # Case 3: Many sequences that barely fit (near max token limit)
        tight_fit = [120000] * 10
        stats3 = self.measure_load_distribution(tight_fit, 130000, 4)
        print(f"Tight fit - Total minibatches: {stats3['total_minibatches']}")

        # Case 4: Some sequences exceed max tokens (should be filtered out in collator)
        with_oversized = [50000, 25000, 140000, 30000, 150000, 20000]
        stats4 = self.measure_load_distribution(with_oversized, 130000, 4)
        print(f"With oversized - Total minibatches: {stats4['total_minibatches']}")

    def test_realistic_outlier_scenarios(self):
        """Test more realistic outlier scenarios that occur in practice."""
        print("\n=== REALISTIC OUTLIER SCENARIOS ===")

        # Case 1: Bimodal distribution (common in instruction/response pairs)
        short_instructions = [np.random.randint(100, 500) for _ in range(100)]
        long_responses = [np.random.randint(50000, 90000) for _ in range(100)]
        bimodal = short_instructions + long_responses
        np.random.shuffle(bimodal)

        stats1 = self.measure_load_distribution(bimodal, 130000, 8)
        print(f"\nBimodal distribution:")
        print(f"  Avg row balance ratio: {stats1['avg_row_balance_ratio']:.3f}")
        print(f"  Total minibatches: {stats1['total_minibatches']}")
        print(f"  Overall efficiency: {stats1['overall_efficiency']:.3f}")

        # Case 2: Power law distribution (few very long, many short)
        power_law = []
        for i in range(200):
            # Power law: length ~ 1/rank^1.5
            length = int(100000 / ((i + 1) ** 0.7))
            power_law.append(max(100, min(100000, length)))
        np.random.shuffle(power_law)

        stats2 = self.measure_load_distribution(power_law, 130000, 8)
        print(f"\nPower law distribution:")
        print(f"  Avg row balance ratio: {stats2['avg_row_balance_ratio']:.3f}")
        print(f"  Total minibatches: {stats2['total_minibatches']}")
        print(f"  Overall efficiency: {stats2['overall_efficiency']:.3f}")

        # Case 3: Clustered lengths (groups of similar sizes)
        cluster1 = [np.random.randint(1000, 2000) for _ in range(50)]
        cluster2 = [np.random.randint(25000, 35000) for _ in range(30)]
        cluster3 = [np.random.randint(70000, 90000) for _ in range(20)]
        clustered = cluster1 + cluster2 + cluster3
        np.random.shuffle(clustered)

        stats3 = self.measure_load_distribution(clustered, 130000, 8)
        print(f"\nClustered lengths:")
        print(f"  Avg row balance ratio: {stats3['avg_row_balance_ratio']:.3f}")
        print(f"  Total minibatches: {stats3['total_minibatches']}")
        print(f"  Overall efficiency: {stats3['overall_efficiency']:.3f}")

        # Case 4: Extreme outliers (few sequences taking up most of max_tokens)
        normal_seqs = [np.random.randint(5000, 15000) for _ in range(50)]
        extreme_outliers = [125000, 128000, 129000]  # Almost at limit
        extreme = normal_seqs + extreme_outliers
        np.random.shuffle(extreme)

        stats4 = self.measure_load_distribution(extreme, 130000, 8)
        print(f"\nExtreme outliers:")
        print(f"  Avg row balance ratio: {stats4['avg_row_balance_ratio']:.3f}")
        print(f"  Total minibatches: {stats4['total_minibatches']}")
        print(f"  Overall efficiency: {stats4['overall_efficiency']:.3f}")

        return stats1, stats2, stats3, stats4


class TestAlgorithmComparison:
    """Compare greedy vs LPT algorithms side by side."""

    def compare_algorithms(self, batch_lengths, max_tokens, num_ranks=8):
        """Run all algorithms and compare their efficiency metrics."""
        # Measure greedy algorithm
        greedy_stats = self._measure_algorithm(
            batch_lengths_to_minibatches, batch_lengths, max_tokens, num_ranks, "Greedy"
        )

        # Measure LPT algorithm
        lpt_stats = self._measure_algorithm(
            batch_lengths_to_minibatches_lpt,
            batch_lengths,
            max_tokens,
            num_ranks,
            "LPT",
        )

        # Print comparison
        print(f"\n{'Metric':<30} {'Greedy':>12} {'LPT':>12} {'Improvement':>18}")
        print("-" * 75)

        metrics = [
            (
                "Total minibatches",
                "total_minibatches",
                lambda g, l: f"{(g - l) / g * 100:.1f}% fewer",
            ),
            (
                "Avg row balance ratio",
                "avg_row_balance_ratio",
                lambda g, l: f"{(l - g) / g * 100:.1f}% better",
            ),
            (
                "Worst row balance ratio",
                "worst_row_balance_ratio",
                lambda g, l: f"{(l - g) / g * 100:.1f}% better",
            ),
            (
                "Row balance std dev",
                "row_balance_std",
                lambda g, l: f"{(g - l) / g * 100:.1f}% lower",
            ),
            (
                "Overall efficiency",
                "overall_efficiency",
                lambda g, l: f"{(l - g) / g * 100:.1f}% higher",
            ),
            (
                "Execution time (ms)",
                "execution_time_ms",
                lambda g,
                l: f"{(l - g) / g * 100:.1f}% {'faster' if l < g else 'slower'}",
            ),
        ]

        for name, key, improvement_fn in metrics:
            greedy_val = greedy_stats[key]
            lpt_val = lpt_stats[key]
            if isinstance(greedy_val, float):
                print(
                    f"{name:<25} {greedy_val:>12.3f} {lpt_val:>12.3f} {improvement_fn(greedy_val, lpt_val):>15}"
                )
            else:
                print(
                    f"{name:<25} {greedy_val:>12} {lpt_val:>12} {improvement_fn(greedy_val, lpt_val):>15}"
                )

        return greedy_stats, lpt_stats

    def _measure_algorithm(
        self, algorithm_fn, batch_lengths, max_tokens, num_ranks, name
    ):
        """Measure performance with focus on per-minibatch (row) balance."""
        # Time the algorithm execution
        start_time = time.perf_counter()

        # Collect results for all ranks
        all_rank_results = []
        for rank in range(num_ranks):
            rank_result = algorithm_fn(batch_lengths, max_tokens, num_ranks, rank)
            all_rank_results.append(rank_result)

        execution_time = time.perf_counter() - start_time

        # Determine total minibatches (rows)
        total_minibatches = max(len(r) for r in all_rank_results)

        # Analyze each minibatch (row) separately
        minibatch_balance_scores = []
        minibatch_efficiencies = []
        total_tokens_used = 0

        for mb_idx in range(total_minibatches):
            # Collect loads for this row across all columns (ranks)
            row_loads = []

            for rank in range(num_ranks):
                if mb_idx < len(all_rank_results[rank]):
                    rank_mb = all_rank_results[rank][mb_idx]
                    # Calculate load for this rank in this minibatch
                    seq_lengths = [batch_lengths[i] for i in rank_mb if i != -1]
                    row_loads.append(sum(seq_lengths))
                else:
                    row_loads.append(0)

            # Calculate balance for this row
            non_zero_loads = [l for l in row_loads if l > 0]
            if non_zero_loads:
                balance_ratio = min(non_zero_loads) / max(non_zero_loads)
                row_efficiency = sum(row_loads) / (num_ranks * max_tokens)
                minibatch_balance_scores.append(balance_ratio)
                minibatch_efficiencies.append(row_efficiency)
                total_tokens_used += sum(row_loads)

        # Calculate aggregate statistics
        avg_row_balance = (
            np.mean(minibatch_balance_scores) if minibatch_balance_scores else 0
        )
        worst_row_balance = (
            np.min(minibatch_balance_scores) if minibatch_balance_scores else 0
        )
        avg_row_efficiency = (
            np.mean(minibatch_efficiencies) if minibatch_efficiencies else 0
        )
        row_balance_std = (
            np.std(minibatch_balance_scores) if minibatch_balance_scores else 0
        )

        # Overall efficiency
        overall_efficiency = total_tokens_used / (
            total_minibatches * num_ranks * max_tokens
        )

        return {
            "algorithm": name,
            "total_minibatches": total_minibatches,
            "avg_row_balance_ratio": avg_row_balance,
            "worst_row_balance_ratio": worst_row_balance,
            "row_balance_std": row_balance_std,
            "avg_row_efficiency": avg_row_efficiency,
            "overall_efficiency": overall_efficiency,
            "execution_time_ms": execution_time * 1000,
            # For backward compatibility in printing
            "load_balance_ratio": avg_row_balance,
            "efficiency": overall_efficiency,
            "load_std": row_balance_std * 1000,  # Scale for display
        }

    def test_comparison_all_scenarios(self):
        """Run comparison on all test scenarios."""
        print("=" * 80)
        print("ALGORITHM COMPARISON: GREEDY vs LPT")
        print("=" * 80)

        # Test 1: Small batch
        print("\n1. SMALL BATCH TEST")
        batch_lengths = [10000, 25000, 35000, 50000, 60000, 75000, 80000, 90000]
        self.compare_algorithms(batch_lengths, 130000, 4)

        # Test 2: Large realistic batch
        print("\n2. LARGE REALISTIC BATCH")
        np.random.seed(42)
        n_sequences = 500
        short_seqs = np.random.randint(100, 2000, size=int(n_sequences * 0.5))
        medium_seqs = np.random.randint(2000, 20000, size=int(n_sequences * 0.3))
        long_seqs = np.random.randint(20000, 60000, size=int(n_sequences * 0.15))
        very_long_seqs = np.random.randint(60000, 100000, size=int(n_sequences * 0.05))
        lengths = np.concatenate([short_seqs, medium_seqs, long_seqs, very_long_seqs])
        np.random.shuffle(lengths)
        self.compare_algorithms(lengths.tolist(), 130000, 8)

        # Test 3: Bimodal distribution
        print("\n3. BIMODAL DISTRIBUTION")
        short_instructions = [np.random.randint(100, 500) for _ in range(100)]
        long_responses = [np.random.randint(50000, 90000) for _ in range(100)]
        bimodal = short_instructions + long_responses
        np.random.shuffle(bimodal)
        self.compare_algorithms(bimodal, 130000, 8)

        # Test 4: Extreme outliers
        print("\n4. EXTREME OUTLIERS")
        normal_seqs = [np.random.randint(5000, 15000) for _ in range(50)]
        extreme_outliers = [125000, 128000, 129000]
        extreme = normal_seqs + extreme_outliers
        np.random.shuffle(extreme)
        self.compare_algorithms(extreme, 130000, 8)

        # Test 5: Power law distribution
        print("\n5. POWER LAW DISTRIBUTION")
        power_law = []
        for i in range(200):
            length = int(100000 / ((i + 1) ** 0.7))
            power_law.append(max(100, min(100000, length)))
        np.random.shuffle(power_law)
        self.compare_algorithms(power_law, 130000, 8)

    def test_speed_benchmark(self):
        """Dedicated speed benchmark with multiple iterations for accurate timing."""
        print("\n6. SPEED BENCHMARK (Multiple Iterations)")
        print("-" * 50)

        # Generate representative test case
        np.random.seed(42)
        n_sequences = 1000
        short_seqs = np.random.randint(100, 2000, size=int(n_sequences * 0.5))
        medium_seqs = np.random.randint(2000, 20000, size=int(n_sequences * 0.3))
        long_seqs = np.random.randint(20000, 60000, size=int(n_sequences * 0.15))
        very_long_seqs = np.random.randint(60000, 100000, size=int(n_sequences * 0.05))
        lengths = np.concatenate([short_seqs, medium_seqs, long_seqs, very_long_seqs])
        np.random.shuffle(lengths)
        batch_lengths = lengths.tolist()

        max_tokens = 130000
        num_ranks = 8
        num_iterations = 10

        # Benchmark greedy algorithm
        greedy_times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            for rank in range(num_ranks):
                batch_lengths_to_minibatches(batch_lengths, max_tokens, num_ranks, rank)
            greedy_times.append(time.perf_counter() - start_time)

        # Benchmark LPT algorithm
        lpt_times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            for rank in range(num_ranks):
                batch_lengths_to_minibatches_lpt(
                    batch_lengths, max_tokens, num_ranks, rank
                )
            lpt_times.append(time.perf_counter() - start_time)

        # Calculate statistics
        greedy_mean = np.mean(greedy_times) * 1000  # Convert to ms
        greedy_std = np.std(greedy_times) * 1000
        lpt_mean = np.mean(lpt_times) * 1000
        lpt_std = np.std(lpt_times) * 1000

        speedup = greedy_mean / lpt_mean if lpt_mean > 0 else float("inf")

        print(
            f"Test setup: {len(batch_lengths)} sequences, {num_ranks} ranks, {num_iterations} iterations"
        )
        print(f"")
        print(
            f"{'Algorithm':<15} {'Mean (ms)':>12} {'Std Dev (ms)':>15} {'Min (ms)':>12} {'Max (ms)':>12}"
        )
        print("-" * 75)
        print(
            f"{'Greedy':<15} {greedy_mean:>12.2f} {greedy_std:>15.2f} {min(greedy_times) * 1000:>12.2f} {max(greedy_times) * 1000:>12.2f}"
        )
        print(
            f"{'LPT':<15} {lpt_mean:>12.2f} {lpt_std:>15.2f} {min(lpt_times) * 1000:>12.2f} {max(lpt_times) * 1000:>12.2f}"
        )
        print(f"")
        if speedup > 1:
            print(f"🚀 LPT is {speedup:.2f}x FASTER than Greedy")
        elif speedup < 1:
            print(f"⚠️  LPT is {1 / speedup:.2f}x SLOWER than Greedy")
        else:
            print("⚖️  Both algorithms have similar performance")

        return {
            "greedy_mean_ms": greedy_mean,
            "greedy_std_ms": greedy_std,
            "lpt_mean_ms": lpt_mean,
            "lpt_std_ms": lpt_std,
            "speedup_factor": speedup,
        }


if __name__ == "__main__":
    # Run efficiency tests directly
    efficiency_tester = TestBatchingEfficiency()

    print("=== BASELINE EFFICIENCY MEASUREMENTS ===")
    print("Current algorithm: Greedy assignment (assign to least loaded rank)")
    efficiency_tester.test_small_batch_efficiency()
    efficiency_tester.test_large_batch_efficiency()
    efficiency_tester.test_varying_sequence_lengths()
    efficiency_tester.test_edge_cases()
    efficiency_tester.test_realistic_outlier_scenarios()

    print("\n" + "=" * 80 + "\n")

    # Run algorithm comparison
    comparison_tester = TestAlgorithmComparison()
    comparison_tester.test_comparison_all_scenarios()

    # Run dedicated speed benchmark
    comparison_tester.test_speed_benchmark()
