""" Confounded Staggered adoption simulation

T: Number of cols
n: Number of measurements per entry
d: Dimension of measurements
beta: Degree of confounding

"""



T, n, d = 80, 30, 4
beta = [3.9/6, 3.9/5]
kernel = "square"
# eta_pool = np.arange(1, 30, 0.5)/3
eta_pool = np.concatenate( ( np.arange(1, 25, 0.5)/6, np.arange(22, 50, 1.5)/5 ) )
i, t = 0, T - 1

nsim = 30

pools = []
pools_eta = []
for i, row_exp in enumerate(np.arange(5, 9)): 
    print(f"{i}-th iteration")
    N = 2**(row_exp)
    perf_pool, eta_star_pool = np.zeros( nsim ), np.zeros( nsim )

    for sim in tqdm(range(nsim)) :
        Data, Masking, pre_Masking, true_Mean, true_Cov = gendata_s_adopt(N, T, n, d, beta, seed = sim)
        true_Mean_it = true_Mean[i, t, :]
        true_Cov_it = true_Cov[i, t, :, :]

        row_Dissim_vec = np.zeros(N)

        for j in range(N) :
            row_Dissim_vec[j] = row_Metric(i, j, t, Data, Masking, kernel, exc_opt = True)

        eta_star = mmDNN_cv(Data, Masking, kernel, eta_cand = eta_pool)
        eta_star_pool[sim] = eta_star

        hat_mu_it = row_mmDNN(i, t, Data, row_Dissim_vec, Masking, eta = eta_star)
        samples_from_truth = np.random.multivariate_normal( true_Mean_it, true_Cov_it, size = (N*n) )
        perf_pool[sim] = sqmmd_est2( hat_mu_it, samples_from_truth, kernel )
    pools_eta.append(eta_star_pool)
    pools.append(perf_pool)    

# Save perf_pool & eta_star_pool
