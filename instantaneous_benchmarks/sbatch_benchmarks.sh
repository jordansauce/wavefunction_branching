#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00

pip install --upgrade pip
conda create -n branching python=3.12 -y
conda activate branching
pip uninstall wavefunction_branching
make install

printf "\n\n\n################################################################################################################################\n"
printf "\n\n\nRunning generate_test_inputs.py\n\n\n"
python instantaneous_benchmarks/generate_test_inputs.py


printf "\n\n\n################################################################################################################################\n"
printf "\n\n\nRunning benchmark_decompositions.py on the ising quench data\n\n\n"
python instantaneous_benchmarks/benchmark_decompositions.py "directory-ising-evo"


printf "\n\n\n################################################################################################################################\n"
printf "\n\n\nRunning benchmark_decompositions.py on the random quench data\n\n\n"
python instantaneous_benchmarks/benchmark_decompositions.py "directory"
