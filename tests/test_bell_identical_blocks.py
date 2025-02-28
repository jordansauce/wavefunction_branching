# # Check that the refactored functions match the original functions
# from pathlib import Path
# import pandas as pd
# import json
# import wavefunction_branching.decompositions.long_range_bell_orig as bell_orig
# if __name__ == "__main__":
#     # Get the directory of sets of "As" matrices to try to simultaneously block-diagonalize
#     directory_path = r"C:\Users\nadro\Documents\Uni\PhD\Ian\code\WavefunctionBranching\benchmarks\decompositions\block_diagonal\block_diagonal_test_data/directory.json"
#     directory = pd.DataFrame(json.load(open(directory_path, mode='rb')))
#     directory.sort_values('dim_L', inplace=True)
#     directory = directory[directory['t'] >= 2.8]

#     for r in directory.index[:1]:
#         row = directory.loc[r]
#         # Validate the path to the As matrices
#         As_path = row.As_file
#         assert isinstance(As_path, str) and As_path != "null"

#         # Load the As matrices to simultaneously block diagonalize
#         As = np.load(As_path)
#         As = make_square(As, xFast=2)
#         norm_As = As.flatten() @ np.conj(As.flatten())
#         As /= np.sqrt(norm_As)

#         # print('\nNo Classical correlations:')
#         # L_orig, S_orig, R_orig, info_orig = combined_optimization(As, n_attempts_iterative=2, n_iterations_per_attempt=1000, early_stopping=False, maxiter_heuristic=10000, keep_classical_correlations=False)


#         print('\nWith classical correlations:')
#         L_new, S_new, R_new, info_new = combined_optimization(As, n_attempts_iterative=1, n_iterations_per_attempt=500, early_stopping=False, maxiter_heuristic=1000, keep_classical_correlations=True)

# #         # theta_orig = einsum(L_orig, S_orig, R_orig, 'b L l, p l r, b r R -> p L R')
# #         # theta_new  = einsum(L_new, S_new, R_new, 'b L l, p l r, b r R -> p L R')
# #         # norm_As = As.flatten() @ np.conj(As.flatten())
# #         # norm_orig = theta_orig.flatten() @ np.conj(theta_orig.flatten())
# #         # norm_new  = theta_new.flatten()  @ np.conj(theta_new.flatten())
# #         # theta_orig /= np.sqrt(norm_orig)
# #         # theta_new  /= np.sqrt(norm_new)
# #         # print(f'As norm         = {norm_As}')
# #         # print(f'theta_orig norm = {norm_orig}')
# #         # print(f'theta_new norm  = {norm_new}')
# #         # print('')


# #         # purif_kc_orig = einsum(L_orig, S_orig, R_orig, 'b L l, p l r, b r R -> b p L R')
# #         # purif_kc_new = einsum(L_new, S_new, R_new, 'b L l, p l r, b r R -> b p L R')
# #         # purif_orig = einsum(L_orig, S_orig, R_orig, 'bl L l, p l r, br r R -> bl br p L R')
# #         # purif_orig = rearrange(purif_orig, 'bl br p L R -> (bl br) p L R')
# #         # purif_new = einsum(L_new, S_new, R_new, 'bl L l, p l r, br r R -> bl br p L R')
# #         # purif_new = rearrange(purif_new, 'bl br p L R -> (bl br) p L R')

# #         # rho_LM_As = einsum(As, np.conj(As), 'p l r, pc lc r -> p l pc lc')
# #         # rho_LM_orig = einsum(purif_orig, np.conj(purif_orig), 'b p l r, b pc lc r -> p l pc lc')
# #         # rho_LM_new = einsum(purif_new,  np.conj(purif_new),   'b p l r, b pc lc r -> p l pc lc')
# #         # trace_distance_LM_orig = measure.trace_distance(rho_LM_orig, rho_LM_As, normalize=True)
# #         # trace_distance_LM_new = measure.trace_distance(rho_LM_new, rho_LM_As, normalize=True)
# #         # trace_distance_LM_orig_new = measure.trace_distance(rho_LM_orig, rho_LM_new, normalize=True)
# #         # print(f'trace_distance_LM_orig      = {trace_distance_LM_orig:.4E}')
# #         # print(f'trace_distance_LM_new       = {trace_distance_LM_new:.4E}')
# #         # print(f'trace_distance_LM_orig_new  = {trace_distance_LM_orig_new:.4E}')
# #         # print('')


# #         # rho_MR_As = einsum(As, np.conj(As), 'p l r, pc l rc -> p r pc rc')
# #         # rho_MR_orig = einsum(purif_orig, np.conj(purif_orig), 'b p l r, b pc l rc -> p r pc rc')
# #         # rho_MR_new = einsum(purif_new,  np.conj(purif_new),   'b p l r, b pc l rc -> p r pc rc')
# #         # trace_distance_MR_orig = measure.trace_distance(rho_MR_orig, rho_MR_As, normalize=True)
# #         # trace_distance_MR_new = measure.trace_distance(rho_MR_new, rho_MR_As, normalize=True)
# #         # trace_distance_MR_orig_new = measure.trace_distance(rho_MR_orig, rho_MR_new, normalize=True)
# #         # print(f'trace_distance_MR_orig      = {trace_distance_MR_orig:.4E}')
# #         # print(f'trace_distance_MR_new       = {trace_distance_MR_new:.4E}')
# #         # print(f'trace_distance_MR_orig_new  = {trace_distance_MR_orig_new:.4E}')
# #         # print('')


# #         # rho_LM_orig = einsum(purif_kc_orig, np.conj(purif_kc_orig), 'b p l r, b pc lc r -> p l pc lc')
# #         # rho_LM_new = einsum(purif_kc_new,  np.conj(purif_kc_new),   'b p l r, b pc lc r -> p l pc lc')
# #         # trace_distance_LM_orig = measure.trace_distance(rho_LM_orig, rho_LM_As, normalize=True)
# #         # trace_distance_LM_new = measure.trace_distance(rho_LM_new, rho_LM_As, normalize=True)
# #         # trace_distance_LM_orig_new = measure.trace_distance(rho_LM_orig, rho_LM_new, normalize=True)
# #         # print(f'trace_distance_LM_orig_kc      = {trace_distance_LM_orig:.4E}')
# #         # print(f'trace_distance_LM_new_kc       = {trace_distance_LM_new:.4E}')
# #         # print(f'trace_distance_LM_orig_new_kc  = {trace_distance_LM_orig_new:.4E}')
# #         # print('')


# #         # rho_MR_As = einsum(As, np.conj(As), 'p l r, pc l rc -> p r pc rc')
# #         # rho_MR_orig = einsum(purif_kc_orig, np.conj(purif_kc_orig), 'b p l r, b pc l rc -> p r pc rc')
# #         # rho_MR_new = einsum(purif_kc_new,  np.conj(purif_kc_new),   'b p l r, b pc l rc -> p r pc rc')
# #         # trace_distance_MR_orig = measure.trace_distance(rho_MR_orig, rho_MR_As, normalize=True)
# #         # trace_distance_MR_new = measure.trace_distance(rho_MR_new, rho_MR_As, normalize=True)
# #         # trace_distance_MR_orig_new = measure.trace_distance(rho_MR_orig, rho_MR_new, normalize=True)
# #         # print(f'trace_distance_MR_orig_kc      = {trace_distance_MR_orig:.4E}')
# #         # print(f'trace_distance_MR_new_kc       = {trace_distance_MR_new:.4E}')
# #         # print(f'trace_distance_MR_orig_new_kc  = {trace_distance_MR_orig_new:.4E}')
# #         # print('')

# #         # print(f'theta_new vs theta_orig overlap  = {theta_orig.flatten() @ np.conj(theta_new.flatten())}')
# #         # print(f'theta_new vs theta_orig distace  = {np.sum(abs(theta_orig.flatten()-theta_new.flatten())**2):.2E}')
# #         # print(f'theta_orig vs As overlap = {As.flatten() @ np.conj(theta_orig.flatten())}')
# #         # print(f'theta_new  vs As overlap = {As.flatten() @ np.conj(theta_new.flatten())}')
# #         # print('')

# #         # for key in info_orig:
# #         #     print(f'{key}:')
# #         #     print(f'    orig: {info_orig[key]}')
# #         #     print(f'    new:  {info_new[key]}')


# #         # print(f'L_orig: {L_orig.shape}, S_orig: {S_orig.shape}, R_orig: {R_new.shape}')
# #         # print(f'L_new:  {L_new.shape}, S_new:  {S_new.shape}, R_new:  {R_new.shape}')

# #         # print(f'L_new vs L_orig distace = {np.sum(abs(L_orig.flatten()-L_new.flatten())**2)}')
# #         # print(f'S_new vs S_orig distace = {np.sum(abs(S_orig.flatten()-S_new.flatten())**2)}')
# #         # print(f'R_new vs R_orig distace = {np.sum(abs(R_orig.flatten()-R_new.flatten())**2)}')

# #         # assert np.allclose(L_orig, L_new)
# #         # assert np.allclose(S_orig, S_new)
# #         # assert np.allclose(R_orig, R_new)

