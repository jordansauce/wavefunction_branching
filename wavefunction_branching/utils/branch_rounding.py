import numpy as np
from samplics.sampling import SampleSelection
from samplics.utils import SelectMethod


def sampford_mask(sample_size, weights):
    """
    PPS-without-replacement sample of fixed size using Rao–Sampford (Sampford) sampling.

    Sampford sampling is a Probability Proportional to Size (PPS) sampling method
    that selects a fixed number of units without replacement, where each unit's
    inclusion probability is approximately proportional to its weight (measure of size).
    Unlike simple random sampling, units with larger weights are more likely to be
    selected, making it useful for situations where certain items should be prioritized
    based on their size or importance.

    This implementation uses the Rao-Sampford method, which ensures that the
    first-order inclusion probabilities are proportional to the weights while
    maintaining a fixed sample size.

    Inputs:
        sample_size : int
            Number of units to select. Must satisfy 1 <= sample_size <= B.
        weights : array-like of shape (B,)
            Nonnegative weights for each unit. Larger weights correspond to
            higher inclusion probabilities. Also called "measure of size" in
            sampling terminology.

    Output:
        mask : ndarray of shape (B,), dtype=int
            Binary indicator vector where 1 indicates selection.
            Exactly `sample_size` entries are 1, and the rest are 0.
            Selection probabilities are approximately proportional to `weights`.
    """
    weights = np.asarray(weights, dtype=float)
    population_size = weights.shape[0]
    if np.any(weights < 0):
        raise ValueError("All weights must be >= 0.")
    if sample_size < 1 or sample_size > population_size:
        raise ValueError("sample_size must be between 1 and B (population size) inclusive.")
    if np.all(weights == 0):
        raise ValueError("At least one weight must be > 0.")

    # Convert sample_size to Python int (samplics doesn't like numpy int64)
    sample_size = int(sample_size)

    # Population unit labels 0..B-1
    unit_ids = np.arange(population_size)

    # Set up Sampford / Rao–Sampford PPS without replacement
    sampler = SampleSelection(
        method=SelectMethod.pps_rs,  # Rao–Sampford PPS
        wr=False,  # sample without replacement
        strat=False,  # not stratified
    )

    # Run the selection
    # sample_flags: length-B array of 0/1 telling whether each unit was selected
    # hits: list/array of the selected unit IDs
    # probs: first-order inclusion probs for each unit
    sample_flags, hits, probs = sampler.select(
        samp_unit=unit_ids,
        samp_size=sample_size,
        mos=weights,  # "measure of size" = weights
    )

    # Convert to binary mask with dtype int
    mask = np.asarray(sample_flags, dtype=int)

    # Sanity check: exactly sample_size ones
    assert mask.sum() == sample_size, "Sampler did not return the requested sample size."

    return mask


def probabilistic_round_child_budget(max_children: int, probs):
    """
    Randomly distributes max_children probabilistically.

    This function is used to round the child budget for each branch.
    It takes a number of objects to distribute and a list of probabilities for bins to distribute them into.
    It returns a list of integers, one for each bin, indicating the number of objects to put each bin in order to match the probabilities.
    The sum of the list will be equal to max_children.
    The function uses a stochastic rounding method to distribute the objects.
    It attempts to distribute the objects as close to the expected values as possible, using randomness only for the fractional parts.

    Inputs:
        max_children : int
            Number of objects to distribute
        probs : list (or numpy array) of floats
            Children will be distributed according to these probabilities.

    Output:
        child_allocations: numpy array of ints
            This will sum to max_children, and be distributed according to probs
    """
    max_children = int(max_children)

    # Check for all-zero probabilities before normalization
    prob_sum = np.sum(probs)
    if prob_sum == 0 or np.isclose(prob_sum, 0.0):
        raise ValueError("All probabilities are zero. At least one probability must be > 0.")

    # normalize the probabilities
    probs = probs / prob_sum

    # array of (non-integer) expected children per branch
    expected_children = max_children * probs

    # round down those numbers, and make sure they're integers
    floored_children = np.floor(expected_children).astype(int)

    # what's the fractional children on each branch
    fractional_children = expected_children - floored_children

    num_children_to_split = max_children - np.sum(floored_children)

    if num_children_to_split == 0:
        child_allocations = floored_children
    else:
        child_allocations = floored_children + sampford_mask(
            num_children_to_split, fractional_children
        )

    if np.sum(child_allocations) != max_children:
        raise RuntimeError("Didn't get the right child allocations")

    return child_allocations
