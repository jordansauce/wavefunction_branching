"""Make sbatch files for sending jobs to the cluster, running wavefunction_branching/evolve_and_branch_finite.py with different branching methods."""

# Default parameters
import copy
from datetime import datetime

dt_time = datetime.now()
seed = 1000000 * dt_time.year + 10000 * dt_time.month + 100 * dt_time.day


def update_dict_copy(old_dict, new_entries):
    new_dict = copy.deepcopy(old_dict)
    new_dict.update(new_entries)
    return new_dict


params_orig = dict(
    iterative_method="pulling_through",
    graddesc_method=None,
    J=1.0,
    g=2.0,
    chi_max=100,
    chi_to_branch=100,
    n_sites=48,
    BC_MPS="finite",
    dt=0.005,
    N_steps_per_output=5,
    svd_min=1e-7,
    trunc_cut=1e-5,
    evo_method="TEBD",
    branching=True,
    max_branches=400,
    max_trace_distance=1.0,
    max_overlap_error=1.0,
    t_evo=50.0,
    min_time_between_branching_attempts=0.2,
    max_branching_attempts=None,
    stop_before_branching=False,
    maxiter_heuristic=6000,
    necessary_local_truncation_improvement_factor=1.0,
    necessary_global_truncation_improvement_factor=1.0,
    seed=seed,
)

# branch_functions = [
#     (None, None),
#     # ('bell_discard_classical', None),
#     # ('bell_keep_classical', None),
#     ('vertical_svd_micro_bsvd', None),
#     # ('pulling_through', None),

#     # ('bell_discard_classical', 'rho_LM_MR_trace_norm_discard_classical_identical_blocks'),
#     # ('bell_keep_classical', 'rho_LM_MR_trace_norm_identical_blocks'),

#     # ('bell_keep_classical', 'rho_LM_MR_trace_norm'),
#     # ('vertical_svd_micro_bsvd', 'rho_LM_MR_trace_norm'),
#     # ('pulling_through', 'rho_LM_MR_trace_norm'),

#     # ('bell_keep_classical', 'graddesc_global_reconstruction_split_non_interfering'),
#     # # ('vertical_svd_micro_bsvd', 'graddesc_global_reconstruction_split_non_interfering'),
#     # # # ('pulling_through', 'graddesc_global_reconstruction_split_non_interfering'),

#     # ('bell_keep_classical', 'graddesc_global_reconstruction_non_interfering'),
#     # ('vertical_svd_micro_bsvd', 'graddesc_global_reconstruction_non_interfering'),
#     # ('pulling_through', 'graddesc_global_reconstruction_non_interfering'),

#     # ('bell_keep_classical', 'rho_half_LM_MR_trace_norm'),
#     # ('vertical_svd_micro_bsvd', 'rho_half_LM_MR_trace_norm'),
#     # ('pulling_through', 'rho_half_LM_MR_trace_norm'),
# ]


runs = []
for n_sites in [80, 128]:
    for chi_max in [50, 100]:
        runs += [
            dict(
                iterative_method=None,
                graddesc_method=None,
                chi_max=chi_max,
                chi_to_branch=999999,
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="bell_discard_classical",
                graddesc_method=None,
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="bell_discard_classical",
                graddesc_method="rho_LM_MR_trace_norm_discard_classical_identical_blocks",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="bell_keep_classical",
                graddesc_method=None,
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="bell_keep_classical",
                graddesc_method="graddesc_global_reconstruction_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="bell_keep_classical",
                graddesc_method="graddesc_global_reconstruction_split_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="bell_keep_classical",
                graddesc_method="rho_LM_MR_trace_norm_identical_blocks",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="bell_keep_classical",
                graddesc_method="rho_LM_MR_trace_norm",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            # dict(
            #     iterative_method = 'bell_keep_classical',
            #     graddesc_method = 'rho_half_LM_MR_trace_norm',
            #     chi_max = chi_max,
            #     chi_to_branch = chi_max,  # int(chi_max*0.75),
            #     n_sites = n_sites,
            #     t_evo = (0.25*n_sites - 2.0)
            # ),
            # Original threshold methods
            dict(
                iterative_method="bell_original_threshold_discard_classical",
                graddesc_method=None,
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
                necessary_local_truncation_improvement_factor=0.0,
                necessary_global_truncation_improvement_factor=0.0,
            ),
            dict(
                iterative_method="bell_original_threshold_discard_classical",
                graddesc_method="rho_LM_MR_trace_norm_discard_classical_identical_blocks",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
                necessary_local_truncation_improvement_factor=0.0,
                necessary_global_truncation_improvement_factor=0.0,
            ),
            dict(
                iterative_method="bell_original_threshold_keep_classical",
                graddesc_method=None,
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
                necessary_local_truncation_improvement_factor=0.0,
                necessary_global_truncation_improvement_factor=0.0,
            ),
            dict(
                iterative_method="bell_original_threshold_keep_classical",
                graddesc_method="graddesc_global_reconstruction_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
                necessary_local_truncation_improvement_factor=0.0,
                necessary_global_truncation_improvement_factor=0.0,
            ),
            dict(
                iterative_method="bell_original_threshold_keep_classical",
                graddesc_method="graddesc_global_reconstruction_split_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
                necessary_local_truncation_improvement_factor=0.0,
                necessary_global_truncation_improvement_factor=0.0,
            ),
            dict(
                iterative_method="bell_original_threshold_keep_classical",
                graddesc_method="rho_LM_MR_trace_norm_identical_blocks",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
                necessary_local_truncation_improvement_factor=0.0,
                necessary_global_truncation_improvement_factor=0.0,
            ),
            dict(
                iterative_method="bell_original_threshold_keep_classical",
                graddesc_method="rho_LM_MR_trace_norm",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
                necessary_local_truncation_improvement_factor=0.0,
                necessary_global_truncation_improvement_factor=0.0,
            ),
            # Other methods
            dict(
                iterative_method="pulling_through",
                graddesc_method=None,
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="pulling_through",
                graddesc_method="graddesc_global_reconstruction_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="pulling_through",
                graddesc_method="graddesc_global_reconstruction_split_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="pulling_through",
                graddesc_method="rho_LM_MR_trace_norm",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            # dict(
            #     iterative_method = 'pulling_through',
            #     graddesc_method = 'rho_half_LM_MR_trace_norm',
            #     chi_max = chi_max,
            #     chi_to_branch = chi_max,  # int(chi_max*0.75),
            #     n_sites = n_sites,
            #     t_evo = (0.25*n_sites - 2.0)
            # ),
            dict(
                iterative_method="vertical_svd_micro_bsvd",
                graddesc_method=None,
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="vertical_svd_micro_bsvd",
                graddesc_method="graddesc_global_reconstruction_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="vertical_svd_micro_bsvd",
                graddesc_method="graddesc_global_reconstruction_split_non_interfering",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            dict(
                iterative_method="vertical_svd_micro_bsvd",
                graddesc_method="rho_LM_MR_trace_norm",
                chi_max=chi_max,
                chi_to_branch=chi_max,  # int(chi_max*0.75),
                n_sites=n_sites,
                t_evo=(0.25 * n_sites - 2.0),
            ),
            # dict(
            #     iterative_method = 'vertical_svd_micro_bsvd',
            #     graddesc_method = 'rho_half_LM_MR_trace_norm',
            #     chi_max = chi_max,
            #     chi_to_branch = chi_max,  # int(chi_max*0.75),
            #     n_sites = n_sites,
            #     t_evo = (0.25*n_sites - 2.0)
            # ),
        ]


filenames = []
# Make the job files
i = 0
for run in runs:
    params = update_dict_copy(params_orig, run)
    i += 1
    branch_function_name = (
        (str(params["iterative_method"]) + "__" + str(params["graddesc_method"]))
        if params["graddesc_method"] is not None
        else str(params["iterative_method"])
    )

    seed = 1000000 * dt_time.year + 10000 * dt_time.month + 100 * dt_time.day
    params["seed"] = seed

    outstr = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20-00:00:00


"""
    for k, v in params.items():
        outstr += f"{k}={v}\n"
    outstr += f"branch_function_name={branch_function_name}\n"

    outstr += r"""
date=$(date '+%Y-%m-%d')
name=L$n_sites-chi$chi_max-at$chi_to_branch-n$max_branches-f_$branch_function_name
outfolder=runs/$date-$name
mkdir -p $outfolder
cp "$0" "$outfolder"/

"""
    # -m cProfile -s time
    outstr += "python wavefunction_branching/evolve_and_branch_finite.py "
    outstr += " --name=$name "
    outstr += " --outfolder=$outfolder "

    for k in params:
        outstr += f" --{k}=${k} "
    outstr += " 2>&1 | tee $outfolder/out-$date-$name.txt"

    filename = f"runs/run_L{params['n_sites']}-chi{params['chi_max']}-at{params['chi_to_branch']}-n{params['max_branches']}-f_{branch_function_name}.sh"
    filenames.append(filename)

    with open(filename, "w") as f:
        f.write(outstr)

    print("\n\n\n ----------------------------------------------------------------------- ")
    print(f"{filename}: ")
    print(" ----------------------------------------------------------------------- ")
    print(outstr)


# Make a bash file which dispatches all of the jobs at once

sbatch_str = "#!/bin/bash \n\n"
sbatch_str += "# pip uninstall physics-tenpy -y \n"
sbatch_str += "# make install \n"
sbatch_str += "# pip install physics-tenpy \n\n"
for filename in filenames:
    sbatch_str += f"chmod u+r+x {filename} \n"
    sbatch_str += f"sbatch      {filename} \n"

with open("runs/sbatch.sh", "w") as f:
    f.write(sbatch_str)


print("\n\n\n ----------------------------------------------------------------------- ")
print("sbatch_str: ")
print(" ----------------------------------------------------------------------- ")
print(sbatch_str)


# branch_functions =  ['blockdiag_2svals_rho_half'] #['blockdiag_extra_terms']
# n_sites_finite   = [32, 48, 64, 80]
# n_sites_infinite = []#[2]#[2, 4, 8, 16, 32, 64]
# maxs_branches = [100, 300]
# filenames = []

# # Make the job files
# for branch_function_name in branch_functions:
#     params['branch_function_name'] = branch_function_name
#     for BC_MPS in ['finite', 'infinite']:
#         params['BC_MPS'] = BC_MPS
#         n_sites_list = n_sites_finite if BC_MPS=='finite' else n_sites_infinite
#         for n_sites in n_sites_list:
#             params['n_sites'] = n_sites
#             for max_branches in maxs_branches:
#                 params['max_branches'] = max_branches

#                 outstr = """#!/bin/bash
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=1
# #SBATCH --mem-per-cpu=8G
# #SBATCH --time=14-00:00:00

# """
#                 for k, v in params.items():
#                     outstr += f"{k}={v}\n"


#                 outstr += r"""
# date=$(date '+%Y-%m-%d')
# name=$branch_function_name-$BC_MPS-L$n_sites-max_branches$max_branches-chi$chi_max-branchat$chi_to_branch
# outfolder=$(pwd)/$date-$name
# mkdir -p $outfolder
# cp "$0" "$outfolder"/

# """

#                 outstr += "python ./wavefunction_branching/branching_scripts/evolve_and_branch_finite.py "
#                 outstr += " --name=$name "
#                 outstr += " --outfolder=$outfolder "


#                 for k in params:
#                     outstr += f" --{k}=${k} "
#                 outstr += " 2>&1 | tee $outfolder/out-$date-$name.txt"

#                 filename = f"run_{params['branch_function_name']}-{params['BC_MPS']}-L{params['n_sites']}-max_branches{params['max_branches']}-chi{params['chi_max']}-branchat{params['chi_to_branch']}.sh"
#                 filenames.append(filename)

#                 with open(filename, 'w') as f:
#                     f.write(outstr)

#                 print(f'\n\n\n ----------------------------------------------------------------------- ')
#                 print(f'{filename}: ')
#                 print(f' ----------------------------------------------------------------------- ')
#                 print(outstr)


# # Make a bash file which dispatches all of the jobs at once

# sbatch_str = '#!/bin/bash \n\n'
# sbatch_str += "make install \n\n"
# for filename in filenames:
#     sbatch_str += f"chmod u+r+x {filename} \n"
#     sbatch_str += f"sbatch      {filename} \n"

# with open('sbatch.sh', 'w') as f:
#     f.write(sbatch_str)


# print(f'\n\n\n ----------------------------------------------------------------------- ')
# print(f'sbatch_str: ')
# print(f' ----------------------------------------------------------------------- ')
# print(sbatch_str)
