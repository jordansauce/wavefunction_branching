# %%
import random

import numpy as np

# --- Simulation Parameters ---
N_TRIALS = 20000  # Number of times to run the simulation for statistics
PARENT_WEIGHT = 1.0  # Initial weight of the parent node
CHILD_BUDGET = 1  # Max number of children allowed (forcing sampling)

# Define candidate branches: (value, probability q_b)
# Scenario: one high-prob (0.9) low-value (1.0) branch,
# one low-prob (0.1) high-value (10.0) branch
candidate_branches = [{"value": 1.0, "prob": 0.9}, {"value": 10.0, "prob": 0.1}]
# Ensure probabilities sum to 1 (already do)
total_prob = sum(b["prob"] for b in candidate_branches)
assert np.isclose(total_prob, 1.0)

# --- Calculate True Expectation Value (before sampling) ---
# This is the weighted average if no sampling occurred.
true_exp_val = sum(b["value"] * b["prob"] for b in candidate_branches) * PARENT_WEIGHT

# --- Simulate Branching and Sampling (Many Trials) ---
results_current_method = []
results_corrected_method = []

print(f"True Expectation Value: {true_exp_val:.4f}")
print(f"Parent Weight: {PARENT_WEIGHT:.4f}")
print("Candidate Branches:")
for i, b in enumerate(candidate_branches):
    print(f"  Branch {i}: Value={b['value']}, Probability (q_b)={b['prob']:.4f}")
print(f"Child Budget: {CHILD_BUDGET}")
print("-" * 30)

branch_probs = np.array([b["prob"] for b in candidate_branches])
branch_indices = np.arange(len(candidate_branches))

for _ in range(N_TRIALS):
    # --- Sampling Step (like np.random.choice) ---
    # Select ONE branch based on probabilities q_b
    selected_index = np.random.choice(branch_indices, p=branch_probs, size=CHILD_BUDGET)[0]

    selected_branch_info = candidate_branches[selected_index]
    q_b = selected_branch_info["prob"]
    value = selected_branch_info["value"]

    # --- Weight Assignment ---

    # Method 1: Current (Flawed) - Effective weight contribution = value * (P_parent * q_b)
    # Simulates scaling the norm by sqrt(q_b) and using norm^2 as weight.
    weight_current = PARENT_WEIGHT * q_b
    results_current_method.append(value * weight_current)  # Value weighted by P_parent*q_b

    # Method 2: Corrected - Effective weight contribution = value * P_parent
    # Simulates inheriting the full parent weight when sampling occurs.
    weight_corrected = PARENT_WEIGHT
    results_corrected_method.append(value * weight_corrected)  # Value weighted by P_parent

# --- Analyze Results ---
# Calculate the average expectation value from the trials
avg_exp_val_current = np.mean(results_current_method)
avg_exp_val_corrected = np.mean(results_corrected_method)

print(f"\n--- Results after {N_TRIALS} trials ---")
print(f"True Expectation Value          : {true_exp_val:.6f}")
print(f"Avg. Exp. Value (Current Method) : {avg_exp_val_current:.6f} (Biased)")
print(f"Avg. Exp. Value (Corrected Method): {avg_exp_val_corrected:.6f} (Unbiased)")

# --- Analytic Calculation for Verification ---
# Expected value (current) = Sum_b [ P(select b) * (value_b * P_parent * q_b) ]
#                        = P_parent * Sum_b [ q_b * value_b * q_b ]
expected_val_current_calc = PARENT_WEIGHT * sum(
    b["prob"] ** 2 * b["value"] for b in candidate_branches
)
# Expected value (corrected) = Sum_b [ P(select b) * (value_b * P_parent) ]
#                          = P_parent * Sum_b [ q_b * value_b ]
expected_val_corrected_calc = PARENT_WEIGHT * sum(
    b["prob"] * b["value"] for b in candidate_branches
)  # This is just the true_exp_val

print(f"\nAnalytic Expected Value (Current): {expected_val_current_calc:.6f}")
print(f"Analytic Expected Value (Corrected): {expected_val_corrected_calc:.6f}")
# %%


import numpy as np


# --- Helper Function: random_round ---
def random_round(x: float) -> int:
    """Randomly round a number up or down, probabilistically to the closest integer"""
    # Use the random module instance for predictable seeding via random.seed()
    ceil = np.ceil(x)
    floor = np.floor(x)
    # Handle cases where x is already an integer
    if np.isclose(ceil, floor):
        return int(x)
    p = x - floor
    r = random.random()
    return int(floor) if r > p else int(ceil)


# --- Simulation Parameters ---
N_TRIALS = 10000
PARENT_WEIGHT = 1.0
CHILD_BUDGET = 3  # Parent's max_children budget (grandchildren budget to distribute)
SEED = 123  # For reproducibility

# Define candidate branches: (value, probability q_b)
candidate_branches_data = [
    {"value": 1.0, "prob": 0.4},  # Branch 0
    {"value": -1.0, "prob": 0.3},  # Branch 1
    {"value": 5.0, "prob": 0.15},  # Branch 2
    {"value": -5.0, "prob": 0.1},  # Branch 3
    {"value": 0.0, "prob": 0.05},  # Branch 4
]
num_candidates = len(candidate_branches_data)

# Ensure probabilities sum to 1
total_prob_check = sum(b["prob"] for b in candidate_branches_data)
if not np.isclose(total_prob_check, 1.0):
    print(f"Warning: Initial probabilities sum to {total_prob_check}. Renormalizing.")
    for b in candidate_branches_data:
        b["prob"] /= total_prob_check

# Add original index for tracking
for i, b in enumerate(candidate_branches_data):
    b["id"] = i

# --- Calculate True Expectation Value (before sampling) ---
true_exp_val = sum(b["value"] * b["prob"] for b in candidate_branches_data) * PARENT_WEIGHT

# --- Simulate Branching and Sampling (Many Trials) ---
results_current_method_contrib = []
results_corrected_method_contrib = []
survival_counts = {
    i: 0 for i in range(num_candidates)
}  # Track how often each original branch survives

print(f"True Expectation Value: {true_exp_val:.6f}")
print(f"Parent Weight: {PARENT_WEIGHT:.4f}")
print(f"Child (Grandchild) Budget: {CHILD_BUDGET}")
print("Candidate Branches:")
for b in candidate_branches_data:
    print(f"  Branch {b['id']}: Value={b['value']}, Probability (q_b)={b['prob']:.4f}")
print("-" * 30)

candidate_probs = np.array([b["prob"] for b in candidate_branches_data])
candidate_indices = np.arange(num_candidates)

# Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

for trial in range(N_TRIALS):
    # --- Stage 1 Sampling: np.random.choice if budget < num_candidates ---
    if num_candidates > CHILD_BUDGET:
        # Sample CHILD_BUDGET indices based on their probabilities
        stage1_kept_indices = np.random.choice(
            candidate_indices,
            p=candidate_probs,
            replace=False,
            size=CHILD_BUDGET,
        )
    else:
        # Keep all candidates if budget allows
        stage1_kept_indices = candidate_indices

    num_selected_stage1 = len(stage1_kept_indices)
    if num_selected_stage1 == 0:  # Should not happen if candidates exist
        continue

    # --- Stage 2 Sampling: Allocate grandchild budget using random_round ---
    selected_branch_probs = candidate_probs[stage1_kept_indices]
    # Rescale probabilities for budget allocation weights
    selected_total_prob = selected_branch_probs.sum()
    if np.isclose(selected_total_prob, 0.0):
        # Avoid division by zero if only zero-prob branches were selected (unlikely)
        allocated_budgets = np.zeros(num_selected_stage1, dtype=int)
    else:
        selected_branch_probs_rescaled = selected_branch_probs / selected_total_prob
        weights_for_rounding = CHILD_BUDGET * selected_branch_probs_rescaled

        # --- Apply random_round with sum correction (simplified version) ---
        allocated_budgets = np.array([random_round(w) for w in weights_for_rounding])
        # Correct sum
        diff = CHILD_BUDGET - allocated_budgets.sum()
        attempts = 0  # Safeguard
        while diff != 0 and attempts < 2 * CHILD_BUDGET:
            adj_idx = np.random.choice(num_selected_stage1)  # Index within the *selected* group
            if diff > 0:
                allocated_budgets[adj_idx] += 1
                diff -= 1
            elif diff < 0 and allocated_budgets[adj_idx] > 0:
                allocated_budgets[adj_idx] -= 1
                diff += 1
            attempts += 1
        if diff != 0:
            # Fallback if correction fails (should be rare with good random_round)
            # print(f"Warning: Budget allocation correction failed on trial {trial}. Diff={diff}")
            pass  # Continue with potentially slightly incorrect total budget

    # Filter based on allocated budget
    stage2_keep_mask = allocated_budgets > 0
    final_survivor_original_indices = stage1_kept_indices[stage2_keep_mask]

    # --- Calculate contributions for this trial ---
    final_survivors = [candidate_branches_data[i] for i in final_survivor_original_indices]
    num_survivors = len(final_survivors)

    trial_contrib_flawed = 0.0
    trial_contrib_corrected = 0.0

    if num_survivors > 0:
        # Track survivors
        for survivor_idx in final_survivor_original_indices:
            survival_counts[survivor_idx] += 1

        # Check if any sampling/filtering occurred compared to the initial candidates
        sampling_occurred = num_candidates > num_survivors

        probs_survived = np.array([b["prob"] for b in final_survivors])
        total_prob_survived = probs_survived.sum()

        # Calculate flawed contribution for the trial
        # weight = P_parent * q_b for each survivor
        trial_contrib_flawed = sum(
            b["value"] * (PARENT_WEIGHT * b["prob"]) for b in final_survivors
        )

        # Calculate corrected contribution for the trial
        if np.isclose(total_prob_survived, 0.0):
            trial_contrib_corrected = 0.0  # Avoid division by zero
        else:
            if sampling_occurred:
                # weight = P_parent * (q_b / total_prob_survived)
                trial_contrib_corrected = sum(
                    b["value"] * (PARENT_WEIGHT * b["prob"] / total_prob_survived)
                    for b in final_survivors
                )
            else:
                # No sampling/filtering occurred, all original branches survived
                # weight = P_parent * q_b
                trial_contrib_corrected = sum(
                    b["value"] * (PARENT_WEIGHT * b["prob"]) for b in final_survivors
                )
                # This should be equal to the true_exp_val for this trial

    results_current_method_contrib.append(trial_contrib_flawed)
    results_corrected_method_contrib.append(trial_contrib_corrected)


# --- Analyze Results ---
avg_exp_val_current = np.mean(results_current_method_contrib)
avg_exp_val_corrected = np.mean(results_corrected_method_contrib)

print(f"\n--- Results after {N_TRIALS} trials ---")
print(f"True Expectation Value             : {true_exp_val:.6f}")
print(f"Avg. Exp. Value (Current Method)   : {avg_exp_val_current:.6f} (Biased)")
print(f"Avg. Exp. Value (Corrected Method) : {avg_exp_val_corrected:.6f} (Unbiased)")

print("\nBranch Survival Frequencies:")
for i in range(num_candidates):
    print(
        f"  Branch {i} survived: {survival_counts[i]/N_TRIALS:.4f} (Prob: {candidate_branches_data[i]['prob']:.4f})"
    )

# --- Analytic Calculation (Approximation) ---
# Calculating the exact analytic expectation after both sampling stages is complex
# because the random_round step couples the selections.
# However, the principle remains: the 'Current Method' weights survivors by q_b
# *after* a q_b-biased selection, while the 'Corrected Method' redistributes
# the parent weight among survivors proportionally. We expect the corrected
# method's average to match the true value.

print(
    "\nNote: Analytic calculation for the current (flawed) method with two-stage",
    "\nsampling and random rounding is complex. The simulation demonstrates the bias.",
)
print("The corrected method should converge to the true expectation value.")
