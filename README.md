# wavefunction_branching
A package for identifying non-interfering "branch" decompositions of Matrix Product State wavefunctions, and then continuing time-evolution after sampling from the branches 

## Installation 
In the terminal, `cd` into this repo then run `make install` or `pip install -e .`


## Running time-evoution with branching
The main script for evolving and branching a wavefunction is located at [wavefunction_branching/evolve_and_branch_finite.py](https://github.com/jordansauce/wavefunction_branching/blob/main/wavefunction_branching/evolve_and_branch_finite.py)

This script can be run using a bash command such as 
```
python wavefunction_branching/evolve_and_branch_finite.py  --name=$name  --outfolder=$outfolder  --J=$J  --g=$g  --chi_max=$chi_max  --n_sites=$n_sites  --BC_MPS=$BC_MPS  --dt=$dt  --N_steps_per_output=$N_steps_per_output  --svd_min=$svd_min  --trunc_cut=$trunc_cut  --evo_method=$evo_method  --branching=$branching  --max_branches=$max_branches  --chi_to_branch=$chi_to_branch  --max_trace_distance=$max_trace_distance  --max_overlap_error=$max_overlap_error  --t_evo=$t_evo  --min_time_between_branching_attempts=$min_time_between_branching_attempts  --max_branching_attempts=$max_branching_attempts  --branch_function_name=$branch_function_name  --stop_before_branching=$stop_before_branching  --maxiter_heuristic=$maxiter_heuristic  --tolEntropy=$tolEntropy  --tolNegativity=$tolNegativity  --seed=$seed  2>&1 | tee $outfolder/out-$date-$name.txt
```
This will produce plots of expectation values, as well as pickle files, stored in `outfolder`. 

Alternatively, you can run `./make_sbatch_files.py` to generate a series of bash files in a `runs` folder. These bash files run simulations reproducing the results of our paper. 


## Analysing results of branching time-evolutions
Then, once multiple runs are complete, you can compare them with two scripts: [evolution_analysis/pickle_analysis.py](https://github.com/jordansauce/wavefunction_branching/blob/main/evolution_analysis/pickle_analysis.py) and [evolution_analysis/error_analysis.py](https://github.com/jordansauce/wavefunction_branching/blob/main/evolution_analysis/error_analysis.py). 
These should be run one after another. The former opens the resulting pickle files matching a pattern, and compiles a dataframe of expectation values and errors over time. The latter should be run after the former, and it plots the errors from the ground-truth in the Ising model.


## Branch decomposition functions
The branching evolution script can use one of many algorithms for performing the branch decompositions. 
To see which branching functions are implemented, check the `iterative_method` and `graddesc_method` variables of the `branch()` and `branch_from_theta()` functions in [wavefunction_branching/decompositions/decompositions.py](https://github.com/jordansauce/wavefunction_branching/blob/main/wavefunction_branching/decompositions/decompositions.py)

```
    iterative_method: None
    | Literal[
        "bell_discard_classical",
        "bell_keep_classical",
        "vertical_svd_micro_bsvd",
        "pulling_through",
    ],
    graddesc_method: None
    | Literal[
        "rho_LM_MR_trace_norm_discard_classical_identical_blocks",
        "rho_LM_MR_trace_norm_identical_blocks",
        "rho_LM_MR_trace_norm",
        # "rho_half_LM_MR_trace_norm",
        "graddesc_global_reconstruction_non_interfering",
        "graddesc_global_reconstruction_split_non_interfering",
    ],
```

These branch functions are defined in [wavefunction_branching/decompositions](https://github.com/jordansauce/wavefunction_branching/tree/main/wavefunction_branching/decompositions).

## Benchmarking branch decomposition functions
Code for benchmarking the branch functions (without needing to run a full evolution) is in [instantaneous_benchmarks](https://github.com/jordansauce/wavefunction_branching/tree/main/instantaneous_benchmarks). There are three relevant files, which should be run in order:
1. [instantaneous_benchmarks/generate_test_inputs.py](https://github.com/jordansauce/wavefunction_branching/blob/main/instantaneous_benchmarks/generate_test_inputs.py) Which generates various inupt tensors to test the decomposition method on
2. [instantaneous_benchmarks/benchmark_decompositions.py](https://github.com/jordansauce/wavefunction_branching/blob/main/instantaneous_benchmarks/benchmark_decompositions.py) Which performs benchmarks on the methods defined within `get_blockdiag_methods()'
3. [instantaneous_benchmarks/plot_benchmarks.py](https://github.com/jordansauce/wavefunction_branching/blob/main/instantaneous_benchmarks/plot_benchmarks.py) Which plots the results of the benchmarks.
