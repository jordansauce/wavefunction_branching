import numpy as np
import pytest

from wavefunction_branching.utils.branch_rounding import (
    probabilistic_round_child_budget,
    sampford_mask,
)


class TestSampfordMask:
    """Tests for sampford_mask function."""

    def test_basic_functionality(self):
        """Test that sampford_mask returns correct sample size."""
        # Use more balanced weights to avoid CertaintyError from samplics
        weights = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        sample_size = 2
        result = sampford_mask(sample_size, weights)

        assert result.sum() == sample_size
        assert result.dtype == int or result.dtype == np.int64 or result.dtype == np.int32
        assert len(result) == len(weights)
        assert np.all((result == 0) | (result == 1))

    def test_large_sample_size(self):
        """Test with large sample size relative to population."""
        # Note: samplics raises CertaintyError when sample_size == population_size or when certain items become certainties.
        # Use uniform weights to avoid certainties
        weights = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        sample_size = 3
        result = sampford_mask(sample_size, weights)

        assert result.sum() == sample_size
        assert len(result) == len(weights)

    def test_single_selection(self):
        """Test when sample_size equals 1."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        sample_size = 1
        result = sampford_mask(sample_size, weights)

        assert result.sum() == sample_size
        assert np.sum(result == 1) == 1

    def test_zero_weights_error(self):
        """Test that all zero weights raise ValueError."""
        weights = np.array([0.0, 0.0, 0.0])
        sample_size = 1

        with pytest.raises(ValueError, match="At least one weight"):
            sampford_mask(sample_size, weights)

    def test_negative_weights_error(self):
        """Test that negative weights raise ValueError."""
        weights = np.array([1.0, -1.0, 2.0])
        sample_size = 1

        with pytest.raises(ValueError, match="All weights"):
            sampford_mask(sample_size, weights)

    def test_invalid_sample_size_error(self):
        """Test that invalid sample_size values raise ValueError."""
        weights = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="sample_size must be between"):
            sampford_mask(0, weights)

        with pytest.raises(ValueError, match="sample_size must be between"):
            sampford_mask(4, weights)

    def test_proportional_to_weights(self):
        """Test that selection probabilities are roughly proportional to weights."""
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_size = 2
        n_trials = 10000

        # Count how many times each element is selected
        counts = np.zeros(len(weights))
        for _ in range(n_trials):
            result = sampford_mask(sample_size, weights)
            counts += result

        # Normalize to get empirical probabilities
        empirical_probs = counts / n_trials

        # Expected probabilities should be roughly proportional to weights
        # Since we're sampling 2 out of 5, the inclusion probabilities are not
        # simply weights / sum(weights), but should still correlate with weights
        # Higher weights should have higher empirical probability
        assert empirical_probs[4] > empirical_probs[0], (
            "Larger weight should have higher selection probability"
        )
        assert empirical_probs[3] > empirical_probs[1], (
            "Larger weight should have higher selection probability"
        )

    def test_uniform_weights(self):
        """Test with uniform weights."""
        weights = np.ones(5)
        sample_size = 2

        # With uniform weights, selection should be roughly uniform
        result = sampford_mask(sample_size, weights)
        assert result.sum() == sample_size


class TestProbabilisticRoundChildBudget:
    """Tests for probabilistic_round_child_budget function."""

    def test_basic_functionality(self):
        """Test basic functionality - sum equals max_children."""
        max_children = 10
        probs = np.array([0.1, 0.2, 0.3, 0.4])

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)
        assert len(result) == len(probs)

    def test_exact_integer_case(self):
        """Test when expected values are exact integers (no rounding needed)."""
        max_children = 10
        probs = np.array([0.2, 0.3, 0.5])  # Expected: [2.0, 3.0, 5.0]

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        np.testing.assert_array_equal(result, [2, 3, 5])

    def test_fractional_case(self):
        """Test when expected values have fractional parts."""
        max_children = 10
        probs = np.array([0.15, 0.25, 0.6])  # Expected: [1.5, 2.5, 6.0]

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        # Floor parts: [1, 2, 6] = 9, so 1 child needs to be allocated
        # Should be allocated probabilistically based on fractional parts [0.5, 0.5, 0.0]
        assert result[0] >= 1 and result[0] <= 2
        assert result[1] >= 2 and result[1] <= 3
        assert result[2] == 6

    def test_floor_deterministic(self):
        """Test that the floor of each allocation is deterministic and matches expected."""
        max_children = 17
        probs = np.array([0.15, 0.23, 0.31, 0.31])  # Expected: [2.55, 3.91, 5.27, 5.27]
        n_trials = 100

        # Calculate expected floors (the deterministic part)
        expected_children = max_children * probs
        expected_floors = np.floor(expected_children).astype(int)
        # Expected floors: [2, 3, 5, 5] = 15, so 2 children need probabilistic allocation

        # Collect all results to verify floor determinism
        all_results = []
        for _ in range(n_trials):
            result = probabilistic_round_child_budget(max_children, probs)
            all_results.append(result)
            # Each result should be at least its expected floor
            assert np.all(result >= expected_floors), (
                f"Result {result} should all be >= expected floors {expected_floors}"
            )
            # Each result should be no more than its expected floor + 1
            assert np.all(result <= expected_floors + 1), (
                f"Result {result} should all be <= expected floors + 1 {expected_floors + 1}"
            )

        all_results = np.array(all_results)

        # Verify the expected floor values explicitly
        # These are the deterministic guarantees: each result will be at least this value
        np.testing.assert_array_equal(expected_floors, np.array([2, 3, 5, 5]))

    def test_uniform_probabilities(self):
        """Test with uniform probabilities."""
        max_children = 12
        probs = np.array([1.0, 1.0, 1.0, 1.0])  # Will be normalized to [0.25, 0.25, 0.25, 0.25]

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        # With uniform probabilities, should be roughly uniform distribution
        assert np.all(result >= 2)  # At least floor(12/4) = 3 each, but fractional parts
        assert np.all(result <= 4)

    def test_single_branch(self):
        """Test with single branch (trivial case)."""
        max_children = 5
        probs = np.array([1.0])

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        assert result[0] == max_children

    def test_large_imbalance(self):
        """Test with highly imbalanced probabilities."""
        max_children = 10
        probs = np.array([0.01, 0.01, 0.98])  # One branch dominates

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        # The large probability branch should get most children
        assert result[2] >= 8  # Should get at least floor(10*0.98) = 9 or 10

    def test_proportional_allocation(self):
        """Test that allocations are roughly proportional to probabilities over many trials."""
        max_children = 100
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        n_trials = 10000

        # Average allocations over many trials
        avg_allocations = np.zeros(len(probs))
        for _ in range(n_trials):
            result = probabilistic_round_child_budget(max_children, probs)
            avg_allocations += result

        avg_allocations /= n_trials

        # Normalize to compare with expected probabilities
        expected = max_children * probs
        # Average should be close to expected (within reasonable tolerance)
        np.testing.assert_allclose(avg_allocations, expected, rtol=0.01)

    def test_zero_probabilities(self):
        """Test handling of zero probabilities (should be normalized out)."""
        max_children = 10
        probs = np.array([0.0, 0.3, 0.7, 0.0])

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        assert result[0] == 0
        assert result[3] == 0
        assert result[1] + result[2] == max_children

    def test_all_zero_probabilities_error(self):
        """Test that all zero probabilities raise ValueError."""
        max_children = 10
        probs = np.array([0.0, 0.0, 0.0])

        # Should raise ValueError before attempting normalization
        with pytest.raises(ValueError, match="All probabilities are zero"):
            probabilistic_round_child_budget(max_children, probs)

    def test_large_max_children(self):
        """Test with large max_children value."""
        max_children = 1000
        probs = np.array([0.15, 0.25, 0.35, 0.25])

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        # Should be roughly proportional
        expected = max_children * probs
        np.testing.assert_allclose(result, expected, rtol=0.01)

    def test_small_max_children(self):
        """Test with small max_children value."""
        max_children = 3
        probs = np.array([0.2, 0.3, 0.5])

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        # With small values, fractional parts matter more
        assert np.all(result >= 0)
        assert np.all(result <= max_children)

    def test_list_input(self):
        """Test that function accepts list input (not just numpy array)."""
        max_children = 10
        probs = [0.1, 0.2, 0.3, 0.4]

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == max_children
        assert isinstance(result, np.ndarray)

    def test_float_max_children(self):
        """Test that function accepts float max_children (should be cast to int)."""
        max_children = 10.7  # Should be cast to 10
        probs = np.array([0.2, 0.3, 0.5])

        result = probabilistic_round_child_budget(max_children, probs)

        assert result.sum() == 10  # Should use int(10.7) = 10
