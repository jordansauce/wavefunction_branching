# wavefunction_branching
A package for identifying non-interfering "branch" decompositions of Matrix Product State wavefunctions, and then continuing time-evolution after sampling from the branches 


## Running time-evoution with branching
The main script for evolving and branching a wavefunction is located at [wavefunction_branching/branching_scripts/evolve_and_branch_finite.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/wavefunction_branching/branching_scripts/evolve_and_branch_finite.py)

This script can be run using a bash command such as 
```
python ./wavefunction_branching/branching_scripts/evolve_and_branch_finite.py  --name=$name  --outfolder=$outfolder  --J=$J  --g=$g  --chi_max=$chi_max  --n_sites=$n_sites  --BC_MPS=$BC_MPS  --dt=$dt  --N_steps_per_output=$N_steps_per_output  --svd_min=$svd_min  --trunc_cut=$trunc_cut  --evo_method=$evo_method  --branching=$branching  --max_branches=$max_branches  --chi_to_branch=$chi_to_branch  --max_trace_distance=$max_trace_distance  --max_overlap_error=$max_overlap_error  --t_evo=$t_evo  --min_time_between_branching_attempts=$min_time_between_branching_attempts  --max_branching_attempts=$max_branching_attempts  --branch_function_name=$branch_function_name  --stop_before_branching=$stop_before_branching  --maxiter_heuristic=$maxiter_heuristic  --tolEntropy=$tolEntropy  --tolNegativity=$tolNegativity  --seed=$seed  2>&1 | tee $outfolder/out-$date-$name.txt
```
This will produce plots of expectation values, as well as pickle files, stored in `outfolder`. 


## Analysing results of branching time-evolutions
Then, once multiple runs are complete, you can compare them with two scripts: [pickle_analysis/pickle_analysis.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/pickle_analysis/pickle_analysis.py) and [pickle_analysis/error_analysis.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/pickle_analysis/error_analysis.py). 
These should be run one after another. The former opens the resulting pickle files matching a pattern, and makes combined plots of expectation values over time. The latter should be run after the former, and it plots the errors from the ground-truth in the Ising model.


## Branch decomposition functions
The branching evolution script can use one of many algorithms for performing the branch decompositions. 
To see which branching functions are implemented, check the ``branch_functions'' variable in the main() function of [evolve_and_branch_finite.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/wavefunction_branching/branching_scripts/evolve_and_branch_finite.py)

```
    branch_functions = {
        None: None,
        'bell': partial(bell.branch, tolEntropy=tolEntropy, tolNegativity=tolNegativity, maxiter_heuristic=maxiter_heuristic,keep_classical_correlations=False),
        'bell_keep_classical': partial(bell.branch, tolEntropy=tolEntropy, tolNegativity=tolNegativity, maxiter_heuristic=maxiter_heuristic,keep_classical_correlations=True),
        'blockdiag_minimal_modification': partial(bmm.branch, tolEntropy=tolEntropy, tolNegativity=tolNegativity, maxiter_heuristic=maxiter_heuristic),
        'blockdiag_2svals_rho_half': partial(bmm_2svals_rho_half.branch, tolEntropy=tolEntropy, tolNegativity=tolNegativity, maxiter_heuristic=maxiter_heuristic)
        }
```

These branch functions are defined in [wavefunction_branching/decompositions](https://github.com/jordansauce/WavefunctionBranching/tree/main/wavefunction_branching/decompositions), as well as many more which are still being benchmarked and haven't yet been added to [evolve_and_branch_finite.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/wavefunction_branching/branching_scripts/evolve_and_branch_finite.py).

## Benchmarking branch decomposition functions
Code for benchmarking the branch functions (without needing to run a full evolution) is in [benchmarks/decompositions](https://github.com/jordansauce/WavefunctionBranching/tree/main/benchmarks/decompositions). There are three relevant files, which should be run in order:
1. [benchmarks/decompositions/block_diagonal/generate_test_inputs.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/benchmarks/decompositions/block_diagonal/generate_test_inputs.py) Which generates various inupt tensors to test the decomposition method on
2. [benchmarks/decompositions/block_diagonal/benchmark_methods_general.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/benchmarks/decompositions/block_diagonal/benchmark_methods_general.py) Which performs benchmarks on the methods defined within `get_blockdiag_methods()'
3. [benchmarks/decompositions/block_diagonal/plot_benchmarks.py](https://github.com/jordansauce/WavefunctionBranching/blob/main/benchmarks/decompositions/block_diagonal/plot_benchmarks.py) Which plots the results of the benchmarks
