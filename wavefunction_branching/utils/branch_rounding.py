import numpy as np
from samplics.sampling import SampleSelection
from samplics.utils import SelectMethod

def sampford_mask(f, M, rng_seed=None):
    """
    PPS-without-replacement sample of fixed size M using Rao–Sampford (Sampford) sampling.
    Inputs:
        f : array-like of shape (B,)
            Nonnegative "size" weights f[b]. Larger f[b] -> higher inclusion probability.
        M : int
            Desired sample size (number of 1s). Must satisfy 1 <= M <= B.
        rng_seed : int or None
            Seed for reproducibility (passed through to samplics if supported).
    Output:
        s : ndarray of shape (B,), dtype=int
            Binary indicator vector.
            Exactly M entries are 1.
            Inclusion probabilities are ~proportional to f.
    """
    f = np.asarray(f, dtype=float)
    B = f.shape[0]
    if np.any(f < 0):
        raise ValueError("All f[b] must be >= 0.")
    if M < 1 or M > B:
        raise ValueError("M must be between 1 and B inclusive.")
    if np.all(f == 0):
        raise ValueError("At least one f[b] must be > 0.")
   
    # population unit labels 0..B-1
    unit_ids = np.arange(B)
   
    # set up Sampford / Rao–Sampford PPS without replacement
    sampler = SampleSelection(
        method=SelectMethod.pps_rs,  # Rao–Sampford PPS
        wr=False,                    # sample without replacement
        strat=False,                 # not stratified
        random_state=rng_seed        # may be ignored on older versions
    )
   
    # run the selection
    # sample_flags: length-B array of 0/1 telling whether each unit was selected
    # hits: list/array of the selected unit IDs
    # probs: first-order inclusion probs for each unit
    sample_flags, hits, probs = sampler.select(
        samp_unit=unit_ids,
        samp_size=M,
        mos=f  # "measure of size" = f[b]
    )
   
    # convert to binary mask s with dtype int
    samp = np.asarray(sample_flags, dtype=int)
   
    # sanity check: exactly M ones
    assert samp.sum() == M, "Sampler did not return the requested sample size."
   
    return samp


def probabilistic_round_child_budget(max_children: int, probs, rng_seed=None):
    """
    Randomly distributes max_children probabalistically. 
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
   
    # normalize the probabilities
    probs = probs/np.sum(probs)
   
    # array of (non-integer) expected children per branch
    expected_children = max_children*probs
   
    # round down those numbers, and make sure they're integers
    floored_children = np.floor(expected_children).astype(int)
   
    # what's the fractional children on each branch
    fractional_children = expected_children - floored_children
   
    num_children_to_split = max_children - np.sum(floored_children)
   
    if num_children_to_split == 0:
        child_allocations = floored_children
    else:
        child_allocations = floored_children + sampford_mask(
            fractional_children,
            num_children_to_split,
            rng_seed=rng_seed  # Pass the rng parameter
        )
   
    if np.sum(child_allocations) != max_children:
        raise RuntimeError("Didn't get the right child allocations")
   
    return child_allocations