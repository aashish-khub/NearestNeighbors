generate_sim_data <- function(N, Ts, rho, seed, sig_noise = 1, snr = NULL) {
  set.seed(seed)
  U <- (runif(N) - 0.5)
  V <- (runif(Ts) - 0.5)

  Y <- abs(outer(U, V, "+")) ^ (rho) * sign(outer(U, V, "+"))
  data_true <- Y

  if(!is.null(snr))
  {
    signal_var = mean(Y^2)
    sig_noise = sqrt(signal_var/snr)
  }

  Y <- Y + sig_noise * matrix(rnorm(N * Ts), nrow = N, ncol = Ts)
  data_noise <- Y

  return(list(
    data_true = data_true,
    data_noise = data_noise,
    row_latent = U,
    col_latent = V,
    noise_sd = sig_noise
  ))
}

mcar <- function(data, seed, missing_prob = 0.5) {
  set.seed(seed)
  N <- dim(data)[1]
  Ts <- dim(data)[2]

  missing_mask <-
    matrix(rbinom(N * Ts, 1, missing_prob),
           nrow = N,
           ncol = Ts)
  missing_mask <- missing_mask == 1

  obs_data <- data
  obs_data[missing_mask] <- NA
  As <- 1 - missing_mask

  return(list(obs_data = obs_data, As = As))
}

mnar <- function(data, seed, obs_prob_mat) {
  set.seed(seed)
  N <- dim(data)[1]
  Ts <- dim(data)[2]

  missing_mask <-
    matrix(rbinom(N * Ts, 1, obs_prob_mat),
           nrow = N,
           ncol = Ts)
  missing_mask <- missing_mask == 0

  obs_data <- data
  obs_data[missing_mask] <- NA
  As <- 1 - missing_mask

  return(list(obs_data = obs_data, As = As))
}


deter_obsv_prob_maker <- function(data, seed, weights = c(0.2, 0.8)) {
  set.seed(seed)
  N <- dim(data)[1]
  Ts <- dim(data)[2]


  # Define the components of the mixture
  component_1 <- rep(0, (N*Ts))                # Degenerate distribution at 0
  component_2 <- ifelse(data > 0, 0.6, 0.4)

  # Randomly select which component each variable should come from
  component_choice <- sample(c(1, 2), size = N*Ts, replace = TRUE, prob = weights)

  # Generate the random variables from the selected components
  probs_from_mix_dist <- matrix(ifelse(component_choice == 1, component_1, component_2), nrow = N, ncol = Ts)
  return(probs_from_mix_dist)
}


synthetic_data_gen <- function(N, Ts, rho, sig_noise, seed = 1, snr = NULL, mode = "mcar", miss_prob = 0.5, mnar_deter = T){
  if(!is.null(snr)){
    data_items <- generate_sim_data(N = N, Ts = Ts, rho = rho, seed = seed, snr = snr)
  }
  else{
    data_items <- generate_sim_data(N = N, Ts = Ts, rho = rho, seed = seed, sig_noise = sig_noise)
  }
  if(grepl(mode, "MCAR", ignore.case = T)){
    observed_data <- mcar(data_items$data_noise, seed, miss_prob)$obs_data
  }
  else if(grepl(mode, "MNAR", ignore.case = T)){
    if(mnar_deter){
      miss_prob_mat <- deter_obsv_prob_maker(data_items$data_true, seed)
    }
    else{
      miss_prob_mat <- deter_obsv_prob_maker(data_items$data_true, seed, weights = c(0,1))
    }
    observed_data <- mnar(data_items$data_noise, seed, miss_prob_mat)$obs_data
  }
  else{
    stop("Error: Mode must be either MCAR or MNAR")
  }
  return(list(observed_data = observed_data,
              observed_entries = !is.na(observed_data),
              full_data_true = data_items$data_true,
              full_data_noise = data_items$data_noise,
              row_latent = data_items$row_latent,
              col_latent = data_items$col_latent,
              noise_sd = data_items$noise_sd))
}

