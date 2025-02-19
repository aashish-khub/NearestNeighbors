""" Confounded Staggered adoption simulation

T: Number of cols
n: Number of measurements per entry
d: Dimension of measurements
beta: Degree of confounding

"""


def gendata_s_adopt(N, T, n, d, beta, seed) : 
    """ Generates Gaussian data, with latent dimension r = 2
    (N, d both be EVEN positive integers)
    Input:
        N: Number of rows
        T: Number of cols
        n: Number of measurements per entry
        d: Dimension of measurements
        beta: vector of two fractions between (0, 1)
    Output:
        
    """
    np.random.seed(seed = seed)

    Data = np.zeros( (N, T, n, d) )
    true_Mean = np.zeros( (N, T, d) )
    true_Cov = np.zeros( (N, T, d, d) )

    u_1 = np.random.uniform(-1, 1, N)
    u_2 = np.random.uniform(0.2, 1, N)

    v_1 = np.random.uniform(-2, 2, T)
    v_2 = np.random.uniform(0.5, 2, T)

    even_ones = np.repeat([0, 1], d/2)
    odd_ones = np.repeat([1, 0], d/2)

    for i in range(N) : 
        for t in range(T) : 
            m_it = u_1[i]*v_1[t]*(even_ones - odd_ones)
            c_it = np.diag(u_2[i]*v_2[t]*(0.5*even_ones + odd_ones))
            true_Mean[i, t, :] = m_it
            true_Cov[i, t, :, :] = c_it
            dat_mat = np.random.multivariate_normal(m_it, c_it, size = n)
            Data[i, t, :, :] = dat_mat

    Masking = np.zeros( (N, T) )
    pre_Masking = np.zeros( (N, T) )

    g1_inds = np.arange(0, N // 3)
    g2_inds = np.arange(N // 3, 2 * N // 3)
    g3_inds = np.arange(2 * N // 3, N)

    gamma_1 = [2, 0.7, 1, 0.7]
    gamma_2 = [2, 0.2, 1, 0.2]

    T1_lower = math.floor(T**beta[0])
    T2_lower = math.floor(T**beta[1])

    for i in range(N) : 
        if i in g1_inds :
            pre_Masking[i, :] = np.concatenate((np.ones(T1_lower), np.zeros(T - T1_lower)))
            for t in range(T - T1_lower) :
                pre_Masking[i, (t + T1_lower)] = np.random.binomial(1, expit(gamma_1[0] + ( 0.99**t )*gamma_1[1]*u_1[i-1] + gamma_1[2]*u_1[i] + ( 0.99**t )*gamma_1[3]*u_1[i+1]), 1)
            pre_A = pre_Masking[i, :]
            if len([i for i in range(len(pre_A)) if pre_A[i] == 0]) == 0:
                Masking[i, :] = pre_A
            elif len([i for i in range(len(pre_A)) if pre_A[i] == 0]) > 0:
                adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0]) 
                Masking[i, :] = np.concatenate((np.ones(adopt_time), np.zeros(T - adopt_time)))
        elif i in g2_inds :
            pre_Masking[i, :] = np.concatenate((np.ones(T2_lower), np.zeros(T - T2_lower)))
            for t in range(T - T2_lower) :
                pre_Masking[i, (t + T2_lower)] = np.random.binomial(1, expit(gamma_2[0] + ( 1.01**t )*gamma_2[1]*u_1[i-1] + gamma_2[2]*u_1[i] + ( 1.01**t )*gamma_2[3]*u_1[i+1]), 1)
            pre_A = pre_Masking[i, :]
            if len([i for i in range(len(pre_A)) if pre_A[i] == 0]) == 0:
                Masking[i, :] = pre_A
            elif len([i for i in range(len(pre_A)) if pre_A[i] == 0]) > 0:
                adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0]) 
                Masking[i, :] = np.concatenate((np.ones(adopt_time), np.zeros(T - adopt_time)))
            # adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0]) 
            # Masking[i, :] = np.concatenate((np.ones(adopt_time), np.zeros(T - adopt_time)))
        elif i in g3_inds : 
            Masking[i, :] = np.ones( T )

    return(Data, Masking, pre_Masking, true_Mean, true_Cov)



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
