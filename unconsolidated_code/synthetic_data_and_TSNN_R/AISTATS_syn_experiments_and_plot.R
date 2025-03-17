mcar_setup <-
  list(
    N_values = c(50, 100, 150, 200),
    rho = 0.5,
    noise_sigma = 0.05,
    snr = 2,
    miss_prob = 0.3
  )
rho <- mcar_setup$rho
mnar_setup <-
  list(
    N_values = c(50, 100, 150, 200),
    rho = 0.5,
    noise_sigma = 0.1,
    snr = 2,
    mnar_deter = T
  )
rho <- mnar_setup$rho
mode <- "MNAR"
seed_collection <- 1:2
res <- algo_comparison_mse_dataset(mode, mcar_setup = mcar_setup, mnar_setup = mnar_setup, seed_collection = seed_collection)
mse_comparison_plotter(res$results, rho, mode)

summary_results <- res$results
filename = paste(mode, " MSE comparison for lambda = ", rho, "over", length(seed_collection), "replications.RData")
save(summary_results, file = filename)



mcar_setup <-
  list(
    N_values = c(50, 100, 150, 200),
    rho_values = c(0.5, 0.75, 1),
    noise_sigma = 1e-4,
    snr = NULL,
    miss_prob = 0.3
  )
mnar_setup <-
  list(
    N_values = c(50, 100, 150, 200),
    rho_values = c(0.6, 0.8, 1),
    noise_sigma = 1e-4,
    snr = NULL,
    mnar_deter = F
  )
mode <- "MNAR"
seed_collection <- 1:8
res_lamb <- mse_sensitivity_lambda_dataset(mode, mcar_setup = mcar_setup, mnar_setup = mnar_setup, seed_collection = seed_collection)
lambda_sensitivity_plotter(res_lamb$results)

summary_results <- res_lamb$results
filename = paste(mode, "MSE sensitivity to lambda over", length(seed_collection), "replications.RData")
save(summary_results, file = filename)
