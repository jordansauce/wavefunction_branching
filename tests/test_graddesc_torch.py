# #%% Non-interfering optimization on random tensors
# if __name__ == '__main__':
#     p = 2
#     L = 50
#     M = 50
#     R = 50
#     T1 = np.random.random((p, L, M))
#     T2 = np.random.random((p, M, R))
#     tensors = normalize([T1, T2])

#     norm = calc_norm(tensors)
#     print(norm)

#     m = 40
#     tensors_a = normalize([np.random.random((p, L, m)), np.random.random((p, m, R))])
#     tensors_b = normalize([np.random.random((p, L, m)), np.random.random((p, m, R))])

#     loss_recon = calc_loss_reconstruction(tensors, tensors_a, tensors_b)
#     print(f'loss_recon = {loss_recon}')

#     tensors_a, tensors_b, info = optimize(tensors, tensors_a, tensors_b, interference_weight=0.1)
#     print(f'norm_a = {calc_norm(tensors_a)}, norm_b = {calc_norm(tensors_b)}')

#     plt.plot(info['losses'], label='total loss')
#     plt.plot(info['reconstruction_losses'], label='reconstruction loss')
#     plt.plot(info['interference_losses'], label = 'interference loss')
#     plt.legend()
#     plt.yscale('log')
#     plt.show()

#     plt.imshow(einsum(abs(contract_theta(to_numpy(tensors_a))), 'p0 p1 m0 m2 -> m0 m2')); plt.show()
#     plt.imshow(einsum(abs(contract_theta(to_numpy(tensors_b))), 'p0 p1 m0 m2 -> m0 m2')); plt.show()

#     #%% Non-interfering optimization on a block diagonal tensor

#     offblock_amount = 1e-6
#     T = np.random.random((p, p, L, R))
#     T[:,:, :L//2, R//2:] *= offblock_amount
#     T[:,:, L//2:, :R//2] *= offblock_amount
#     T = rearrange(T, 'p0 p1 L R -> (p0 L) (p1 R)')
#     T1, s, T2 = np.linalg.svd(T, full_matrices=False)
#     T1 = einsum(T1, s**0.5, 'pL M, M -> pL M')
#     T2 = einsum(T2, s**0.5, 'M pR, M -> M pR')
#     T1 = rearrange(T1, '(p L) M -> p L M', p=p)
#     T2 = rearrange(T2, 'M (p R)-> p M R', p=p)
#     tensors = normalize([T1, T2])

#     tensors_a = normalize([np.random.random((p, L, m)), np.random.random((p, m, R))])
#     tensors_b = normalize([np.random.random((p, L, m)), np.random.random((p, m, R))])
#     tensors_a, tensors_b, info = optimize(tensors, tensors_a, tensors_b, interference_weight=0.05, lr=0.0015, gamma=0.97, epochs=50)
#     print(f'norm_a = {calc_norm(tensors_a)}, norm_b = {calc_norm(tensors_b)}')
#     plt.plot(info['losses'], label='total loss')
#     plt.plot(info['reconstruction_losses'], label='reconstruction loss')
#     plt.plot(info['interference_losses'], label = 'interference loss')
#     plt.legend()
#     plt.yscale('log')
#     plt.show()

#     plt.imshow(einsum(abs(contract_theta(to_numpy(tensors_a))), 'p0 p1 m0 m2 -> m0 m2')); plt.show()
#     plt.imshow(einsum(abs(contract_theta(to_numpy(tensors_b))), 'p0 p1 m0 m2 -> m0 m2')); plt.show()


# %%
