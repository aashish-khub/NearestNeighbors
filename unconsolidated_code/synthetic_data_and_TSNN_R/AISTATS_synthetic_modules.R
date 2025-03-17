library(foreach)
library(doParallel)
library(matrixStats)
library(Rcpp)
library(dplyr)
library(ggplot2)
library(scales)
library(broom)


cppFunction('
NumericMatrix compute_distances_rcpp(NumericMatrix X_obs, bool is_row) {
  int n, m;
  if (is_row) {
    n = X_obs.nrow();
    m = X_obs.ncol();
  } else {
    n = X_obs.ncol();
    m = X_obs.nrow();
  }

  NumericMatrix distances(n, n);

  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      std::vector<double> vec1, vec2;
      for (int k = 0; k < m; ++k) {
        double val1 = is_row ? X_obs(i, k) : X_obs(k, i);
        double val2 = is_row ? X_obs(j, k) : X_obs(k, j);
        if (!NumericVector::is_na(val1) && !NumericVector::is_na(val2)) {
          vec1.push_back(val1);
          vec2.push_back(val2);
        }
      }

      if (!vec1.empty()) {
        double dist_val = 0.0;
        for (size_t k = 0; k < vec1.size(); ++k) {
          dist_val += (vec1[k] - vec2[k]) * (vec1[k] - vec2[k]);
        }
        dist_val /= vec1.size();
        distances(i, j) = dist_val;
        distances(j, i) = dist_val;
      } else {
        distances(i, j) = R_PosInf;
        distances(j, i) = R_PosInf;
      }
    }
  }

  return distances;
}
')

double_nearest_neighbor_oracle <-
  function(valid_data,
           obs_data,
           percentiles,
           row_latent,
           col_latent,
           row_percentiles = NULL,
           self_include = FALSE,
           verbose = FALSE) {
    # row_latent = as.numeric(row_latent)
    # col_latent = as.numeric(col_latent)

    n <- nrow(obs_data)
    m <- ncol(obs_data)

    row_distances = (outer(row_latent, row_latent, "-")) ^ 2
    col_distances = (outer(col_latent, col_latent, "-")) ^ 2
    eta1_values <- quantile(row_distances[row_distances != Inf & row_distances > 0], row_percentiles / 100)
    eta2_values <- quantile(col_distances[col_distances != Inf & col_distances > 0], percentiles / 100)
    nloop <- length(eta1_values)

    results <- NULL

    num_cores <- detectCores()

    results <- mclapply(1:nloop, function(index) {
      intermediate_results <- NULL
      eta1 <- eta1_values[index]
      index_set_of_eta2 <- (max(1, index-2): min(index+2, nloop))
      range_of_eta2 <- as.numeric(eta2_values[index_set_of_eta2])
      for(col_index in 1:length(index_set_of_eta2)) {
        predicted_X <- matrix(NA, n, m)
        for (u in 1:n) {
          for (i in 1:m) {
            if(!self_include) {
              temp <- obs_data[u, i]
              obs_data[u, i] <- NA
            }
            eta2 <- range_of_eta2[col_index]
            row_neighbours <- which(row_distances[u, ] <= eta1)
            col_neighbours <- which(col_distances[i, ] <= eta2)
            values <- obs_data[row_neighbours, col_neighbours]
            observed_values <- values[!is.na(values)]
            if (length(observed_values) > 0) {
              predicted_X[u, i] <- mean(observed_values)
            }
            if(!self_include) {
              obs_data[u,i] <- temp
              rm(temp)
            }
          }
        }
        mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)
        intermediate_results <- rbind(intermediate_results,
                                      data.frame(eta1 = eta1, eta2 = eta2, mse = mse))
      }
      return(intermediate_results)
    }, mc.cores = num_cores - 3)

    results <- do.call(rbind, results)
    # Finding the best MSE and corresponding etas
    best_result <- results[which.min(results$mse),]
    best_eta1 <- best_result$eta1
    best_eta2 <- best_result$eta2
    best_mse <- best_result$mse
    row_dist_cdf <- ecdf(row_distances[row_distances != Inf])
    col_dist_cdf <- ecdf(col_distances[col_distances != Inf])
    eta1_percentile <- row_dist_cdf(best_eta1)
    eta2_percentile <- col_dist_cdf(best_eta2)

    row_neighbours <- list()
    col_neighbours <- list()
    for(u in 1:n) {
      row_neighbours[[u]] <- which(row_distances[u, ] <= best_eta1)
    }
    for(i in 1:m) {
      col_neighbours[[i]] <- which(col_distances[i, ] <= best_eta2)
    }

    cat(
      "TS-NN oracle train MSE:",
      best_mse,
      "\n dim:",
      dim(valid_data) ,
      "\n best eta1 percentile:",
      eta1_percentile,
      "\n best eta2 percentile:",
      eta2_percentile,
      "\n"
    )
    return(
      list(
        eta1 = best_eta1,
        eta2 = best_eta2,
        mse = best_mse,
        eta1_percentile = eta1_percentile,
        eta2_percentile = eta2_percentile,
        row_neighbors_list = row_neighbours,
        col_neighbors_list = col_neighbours
      )
    )
  }

double_nearest_neighbor_oracle_evaluator <-
  function(true_signal, obs_data, row_neighbors_list, col_neighbors_list, verbose = TRUE, self_include = TRUE) {

    predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
    for (u in 1:nrow(obs_data)) {
      for (i in 1:ncol(obs_data)) {
        row_neighbours <- row_neighbors_list[[u]]
        col_neighbours <- col_neighbors_list[[i]]
        if(!self_include) {
          temp <- obs_data[u, i]
          obs_data[u, i] <- NA
        }
        values <- obs_data[row_neighbours, col_neighbours]
        observed_values <- values[!is.na(values)]
        if (length(observed_values) > 0) {
          predicted_X[u, i] <- mean(observed_values)
        }
        if(!self_include) {
          obs_data[u,i] <- temp
          rm(temp)
        }
      }
    }


    eval_mse <- mean((true_signal - predicted_X) ^ 2, na.rm = TRUE)


    cat(
      "TS-NN Oracle test MSE:",
      eval_mse,
      "dim:",
      dim(true_signal) ,
      "\n",
      "\n"
    )
    # You would need to adjust this part to return the best_predicted_X
    return(eval_mse)
  }

double_nearest_neighbour_parallel <- function(valid_data, obs_data, percentiles = c(10, 13, 17, 20), row_percentiles = NULL, self_include = FALSE, verbose = FALSE) {
  best_mse <- Inf
  best_eta1 <- NA
  best_eta2 <- NA
  best_predicted_X <- NA

  n <- nrow(obs_data)
  m <- ncol(obs_data)

  row_distances <- compute_distances_rcpp(obs_data, TRUE)
  col_distances <- compute_distances_rcpp(obs_data, FALSE)

  if(is.null(row_percentiles)) {
    row_percentiles = percentiles
  }

  eta1_values <- quantile(row_distances[row_distances != Inf & row_distances > 0], row_percentiles / 100)
  eta2_values <- quantile(col_distances[col_distances != Inf & col_distances > 0], percentiles / 100)
  nloop <- length(eta1_values)

  # Register parallel backend
  num_cores <- detectCores()

  results <- mclapply(1:nloop, function(index) {
    intermediate_results <- NULL
    eta1 <- eta1_values[index]
    index_set_of_eta2 <- (max(1, index-3): min(index+3, nloop))
    range_of_eta2 <- as.numeric(eta2_values[index_set_of_eta2])
    for(col_index in 1:length(index_set_of_eta2)) {
      predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
      for (u in 1:n) {
        for (i in 1:m) {
          if(!self_include) {
            temp <- obs_data[u, i]
            obs_data[u, i] <- NA
          }
          eta2 <- range_of_eta2[col_index]
          row_neighbours <- which(row_distances[u, ] <= eta1)
          col_neighbours <- which(col_distances[i, ] <= eta2)
          values <- obs_data[row_neighbours, col_neighbours]
          observed_values <- values[!is.na(values)]
          if (length(observed_values) > 0) {
            predicted_X[u, i] <- mean(observed_values)
          }
          if(!self_include) {
            obs_data[u,i] <- temp
            rm(temp)
          }
        }
      }
      mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)
      intermediate_results <- rbind(intermediate_results,
                                    data.frame(eta1 = eta1, eta2 = eta2, mse = mse))
    }
    return(intermediate_results)
  }, mc.cores = num_cores - 1)

  results <- do.call(rbind, results)
  # Finding the best MSE and corresponding etas
  best_result <- results[which.min(results$mse),]
  best_eta1 <- best_result$eta1
  best_eta2 <- best_result$eta2
  best_mse <- best_result$mse
  row_dist_cdf <- ecdf(row_distances[row_distances != Inf])
  col_dist_cdf <- ecdf(col_distances[col_distances != Inf])
  eta1_percentile <- row_dist_cdf(best_eta1)
  eta2_percentile <- col_dist_cdf(best_eta2)
  row_neighbours <- list()
  col_neighbours <- list()
  for(u in 1:n) {
    row_neighbours[[u]] <- which(row_distances[u, ] <= best_eta1)
  }
  for(i in 1:m) {
    col_neighbours[[i]] <- which(col_distances[i, ] <= best_eta2)
  }
  if(verbose) {
    cat(
      "TS-NN train MSE:", best_mse,
      "eta1 percentile", eta1_percentile,
      "eta2 percentile", eta2_percentile,
      "\n"
    )
  }

  return(
    list(
      eta1 = best_eta1,
      eta2 = best_eta2,
      mse = best_mse,
      eta1_percentile = eta1_percentile,
      eta2_percentile = eta2_percentile,
      row_neighbors_list = row_neighbours,
      col_neighbors_list = col_neighbours
    )
  )
}

double_nearest_neighbour_evaluator <-
  function(valid_data, obs_data, row_neighbors_list, col_neighbors_list, verbose = TRUE, self_include = TRUE, true_signal = NULL) {

    predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
    for (u in 1:nrow(obs_data)) {
      for (i in 1:ncol(obs_data)) {
        row_neighbours <- row_neighbors_list[[u]]
        col_neighbours <- col_neighbors_list[[i]]
        if(!self_include) {
          temp <- obs_data[u, i]
          obs_data[u, i] <- NA
        }
        values <- obs_data[row_neighbours, col_neighbours]
        observed_values <- values[!is.na(values)]
        if (length(observed_values) > 0) {
          predicted_X[u, i] <- mean(observed_values)
        }
        if(!self_include) {
          obs_data[u,i] <- temp
          rm(temp)
        }
      }
    }

    if(is.null(true_signal)){
      eval_mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)
    }
    else{
      eval_mse <- mean((true_signal - predicted_X) ^ 2, na.rm = TRUE)
    }

    cat(
      "TS-NN test MSE:",
      eval_mse,
      "dim:",
      dim(valid_data) ,
      "\n","\n"
    )
    # You would need to adjust this part to return the best_predicted_X
    return(eval_mse)
  }

time_nearest_neighbour_parallel <- function(valid_data, obs_data, percentiles = c(10, 13, 17, 20), self_include = FALSE, verbose = TRUE) {
  col_distances <- compute_distances_rcpp(obs_data, FALSE)
  eta2_values <- quantile(col_distances[col_distances != Inf & col_distances > 0], percentiles / 100)

  m <- ncol(obs_data)

  num_cores <- detectCores()

  results <- mclapply(eta2_values, function(eta2) {
    predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
    for (i in 1:m) {
      col_neighbours <- which(col_distances[, i] <= eta2)
      if(!self_include) {
        if (i %in% col_neighbours) {
          col_neighbours <- col_neighbours[-which(col_neighbours == i)]
        }
      }
      if (length(col_neighbours) == 1) {
        predicted_X[, i] <- obs_data[, col_neighbours]
      }
      else if (length(col_neighbours) > 1) {
        predicted_X[, i] <- rowMeans(obs_data[, col_neighbours], na.rm = TRUE)
      }
    }
    mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)

    return(data.frame(eta2 = eta2, mse = mse))
  }, mc.cores = num_cores - 4)

  results <- do.call(rbind, results)

  # Find best result
  best_idx <- which.min(results$mse)
  best_eta2 <- results$eta2[best_idx]
  best_mse <- results$mse[best_idx]

  col_neighbours <- list()
  for(i in 1:m) {
    col_neighbours[[i]] <- which(col_distances[, i] <= best_eta2)
  }

  col_dist_cdf <- ecdf(col_distances[col_distances != Inf])
  eta1_percentile <- col_dist_cdf(best_eta2)

  if (verbose) {
    cat(
      "Time-NN train MSE:", best_mse,
      "Eta2 percentile:", eta1_percentile, "\n"
    )
  }

  return(list(
    eta2 = best_eta2,
    mse = best_mse,
    eta_percentile = eta1_percentile,
    neighbors_list = col_neighbours
  ))
}

time_nearest_neighbour_evaluator <- function(valid_data, obs_data, neighbors_list, verbose = TRUE, self_include = TRUE, true_signal = NULL) {
  predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))

  for (i in 1:ncol(obs_data)) {
    col_neighbours <- neighbors_list[[i]]
    if(!self_include) {
      if (i %in% col_neighbours) {
        col_neighbours <- col_neighbours[-which(col_neighbours == i)]
      }
    }
    if (length(col_neighbours) == 1) {
      predicted_X[, i] <- obs_data[, col_neighbours]
    }
    else if (length(col_neighbours) > 1) {
      predicted_X[, i] <- rowMeans(obs_data[, col_neighbours], na.rm = TRUE)
    }
  }
  mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)

  if (verbose) {
    cat("Time-NN test MSE:",
        mse,
        "dim:",
        dim(valid_data) ,
        "\n")
  }
  return(mse)
}

doubly_robust_nearest_neighbour_parallel <- function(valid_data, obs_data, percentiles = c(10, 13, 17, 20), row_percentiles = NULL, self_include = FALSE, verbose = FALSE) {
  best_mse <- Inf
  best_eta1 <- NA
  best_eta2 <- NA
  best_predicted_X <- NA

  n <- nrow(obs_data)
  m <- ncol(obs_data)

  row_distances <- compute_distances_rcpp(obs_data, TRUE)
  col_distances <- compute_distances_rcpp(obs_data, FALSE)

  if(is.null(row_percentiles)) {
    row_percentiles = percentiles
  }

  eta1_values <-
    quantile(row_distances[row_distances != Inf & row_distances > 0], row_percentiles / 100)
  eta2_values <-
    quantile(col_distances[col_distances != Inf & col_distances > 0], percentiles / 100)

  nloop <- length(eta1_values)

  # Register parallel backend
  num_cores <- detectCores()
  registerDoParallel(cores = num_cores - 3)

  results <-
    foreach(
      index = 1:nloop,
      .combine = rbind,
      .packages = c('matrixStats')
    ) %dopar% {
      eta1 <- eta1_values[index]
      index_set_of_eta2 <- c(max(1, index - 2), min(index + 2, nloop))
      intermediate_results <- NULL
      for(eta2 in eta2_values[index_set_of_eta2]) {
        predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
        eta2 <- eta2_values[index]
        for (u in 1:n) {
          for (i in 1:m) {
            if(!self_include) {
              temp <- obs_data[u, i]
              obs_data[u, i] <- NA
            }
            row_neighbours <- which(row_distances[u, ] <= eta1)
            col_neighbours <- which(col_distances[i, ] <= eta2)

            row_ngd_size <- length(row_neighbours)
            col_ngd_size <- length(col_neighbours)
            row_values <- obs_data[row_neighbours, i]
            row_matrix <- matrix(row_values, nrow = row_ngd_size,
                                 ncol = col_ngd_size, byrow = FALSE)
            col_values <- obs_data[u, col_neighbours]
            col_matrix <- matrix(col_values, nrow = row_ngd_size,
                                 ncol = col_ngd_size, byrow = TRUE)
            values <- obs_data[row_neighbours, col_neighbours]
            if (sum(!is.na(row_values)) > 0 & sum(!is.na(col_values)) > 0) {
              predicted_X[u, i] <- mean(row_matrix + col_matrix - values, na.rm = TRUE)
            }
            if(!self_include) {
              obs_data[u, i] <- temp
              rm(temp)
            }
          }
        }
        mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)
        intermediate_results <- rbind(intermediate_results,
                                      data.frame(eta1 = eta1,
                                                 eta2 = eta2,
                                                 mse = mse))
      }
      return(intermediate_results)
    }

  # Finding the best MSE and corresponding etas
  best_result <- results[which.min(results$mse),]
  best_eta1 <- best_result$eta1
  best_eta2 <- best_result$eta2
  best_mse <- best_result$mse

  row_neighbours <- list()
  col_neighbours <- list()
  for(u in 1:n) {
    row_neighbours[[u]] <- which(row_distances[u, ] <= best_eta1)
  }
  for(i in 1:m) {
    col_neighbours[[i]] <- which(col_distances[i, ] <= best_eta2)
  }

  row_dist_cdf <- ecdf(row_distances[row_distances != Inf])
  col_dist_cdf <- ecdf(col_distances[col_distances != Inf])
  eta1_percentile <- row_dist_cdf(best_eta1)
  eta2_percentile <- col_dist_cdf(best_eta2)

  stopImplicitCluster()
  if(verbose){
    cat(
      "DR-NN MSE:",
      best_mse,
      "eta1 per",
      eta1_percentile,
      "eta2 per",
      eta2_percentile,
      "\n"
    )
  }
  # You would need to adjust this part to return the best_predicted_X
  return(
    list(
      eta1 = best_eta1,
      eta2 = best_eta2,
      mse = best_mse,
      eta1_percentile = eta1_percentile,
      eta2_percentile = eta2_percentile,
      row_neighbors_list = row_neighbours,
      col_neighbors_list = col_neighbours
    )
  )
}

doubly_robust_nearest_neighbour_evaluator <-
  function(valid_data, obs_data, row_neighbors_list, col_neighbors_list, verbose = TRUE, self_include = TRUE, true_signal = NULL) {

    predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
    for (u in 1:nrow(obs_data)) {
      for (i in 1:ncol(obs_data)) {
        row_neighbours <- row_neighbors_list[[u]]
        col_neighbours <- col_neighbors_list[[i]]
        if(!self_include) {
          temp <- obs_data[u, i]
          obs_data[u, i] <- NA
        }
        row_ngd_size <- length(row_neighbours)
        col_ngd_size <- length(col_neighbours)
        row_values <- obs_data[row_neighbours, i]
        row_matrix <- matrix(row_values, nrow = row_ngd_size,
                             ncol = col_ngd_size, byrow = FALSE)
        col_values <- obs_data[u, col_neighbours]
        col_matrix <- matrix(col_values, nrow = row_ngd_size,
                             ncol = col_ngd_size, byrow = TRUE)
        values <- obs_data[row_neighbours, col_neighbours]
        if (sum(!is.na(row_values)) > 0 & sum(!is.na(col_values)) > 0) {
          predicted_X[u, i] <- mean(row_matrix + col_matrix - values, na.rm = TRUE)
        }
        if(!self_include) {
          obs_data[u,i] <- temp
          rm(temp)
        }
      }
    }

    if(is.null(true_signal)){
      eval_mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)
    }
    else{
      eval_mse <- mean((true_signal - predicted_X) ^ 2, na.rm = TRUE)
    }

    cat(
      "DR-NN test MSE:",
      eval_mse,
      "dim:",
      dim(valid_data) ,
      "\n"
    )
    # You would need to adjust this part to return the best_predicted_X
    return(eval_mse)
  }

#USVT
library(filling)
USVT_mc <- function(X_full, X_obs, eta = 0.01) {
  predicted_X = fill.USVT(X_obs, eta = eta)$X
  mse <- mean((X_full - predicted_X) ^ 2, na.rm = TRUE)
  cat("MSE:", mse, "\n")
  return(mse)
}


#SoftImpute
library(softImpute)
soft_impute_eval <- function(obs_data, true_data, rank = 50, lambdas = 13 - (exp(seq(log(1), log(12), length.out = 10))), type = "als", verbose = T){
  rank <- min(rank, min(dim(obs_data)) - 1)
  min_mse <- Inf
  opti_lamb <- NA
  for (lambda in lambdas) {
    fit1 <- softImpute(obs_data, rank = rank, lambda = lambda, type = type)
    soft_imputed_dat <- complete(obs_data, fit1)
    mse <- mean((soft_imputed_dat - true_data)^2, na.rm = TRUE)
    if(mse < min_mse) {
      min_mse <- mse
      opti_lamb <- lambda
    }
  }
  if(verbose) {
    cat("SoftImpute min MSE:", min_mse, " for lambda:", opti_lamb, "\n dim:", dim(obs_data),"\n")
  }

  return(list(mse = min_mse, lambda = opti_lamb))
}




cross_val <-
  function(dataset,
           true_dataset,
           func,
           percentiles,
           eval_func,
           nfolds = 5,
           row_percentiles = NULL,
           two_sided_nn = TRUE,
           verbose = TRUE,
           self_include = FALSE,
           eval_self_include = F) {
    n <- nrow(dataset)
    m <- ncol(dataset)
    temp <- sample((1 * nfolds), n * m, replace = TRUE)
    fold_mat <- matrix(temp, nrow = n)
    cv_results <- rep(NA, nfolds)
    for (fold in 1:nfolds) {
      test_data <-
        matrix(ifelse(fold_mat %in% fold, dataset, NA), ncol = ncol(dataset))
      test_true_signal <-
        matrix(ifelse(fold_mat %in% fold, true_dataset, NA),
               ncol = ncol(dataset))
      train_data <-
        matrix(ifelse(!(fold_mat %in% fold), dataset, NA), ncol = ncol(dataset))
      if (two_sided_nn) {
        fold_res <-
          func(
            train_data,
            train_data,
            percentiles,
            verbose = verbose,
            self_include = self_include,
            row_percentiles = row_percentiles
          )
        cv_results[fold] <-
          eval_func(
            test_data,
            dataset,
            fold_res$row_neighbors_list,
            fold_res$col_neighbors_list,
            self_include = eval_self_include,
            true_signal = test_true_signal
          )
      }
      else{
        fold_res <-
          func(
            train_data,
            train_data,
            percentiles,
            verbose = verbose,
            self_include = self_include
          )
        cv_results[fold] <-
          eval_func(test_data,
                    dataset,
                    fold_res$neighbors_list,
                    self_include = eval_self_include,
                    true_signal = test_true_signal)
      }
    }
    return(list(
      cv_err_mean = mean(cv_results),
      cv_err_sd = sd(cv_results),
      cv_errors = cv_results
    ))
  }

cross_val_tsnn_oracle <-
  function(dataset,
           true_dataset,
           percentiles,
           row_latent,
           col_latent,
           nfolds = 5,
           verbose = FALSE,
           row_percentiles = NULL,
           self_include = FALSE,
           eval_self_include = F) {
    n <- nrow(dataset)
    m <- ncol(dataset)
    temp <- sample((2 * nfolds), n * m, replace = TRUE)
    fold_mat <- matrix(temp, nrow = n)
    cv_results <- rep(NA, nfolds)
    for (fold in 1:nfolds) {
      test_data <-
        matrix(ifelse(fold_mat %in% fold, dataset, NA), ncol = ncol(dataset))
      test_true_signal <-
        matrix(ifelse(fold_mat %in% fold, true_dataset, NA),
               ncol = ncol(dataset))
      train_data <-
        matrix(ifelse(!(fold_mat %in% fold), dataset, NA), ncol = ncol(dataset))
      fold_res <-
        double_nearest_neighbor_oracle(
          train_data,
          train_data,
          percentiles,
          row_percentiles = row_percentiles,
          row_latent = row_latent,
          col_latent = col_latent,
          verbose = verbose,
          self_include = self_include
        )
      cv_results[fold] <-
        double_nearest_neighbor_oracle_evaluator(
          test_true_signal,
          train_data,
          fold_res$row_neighbors_list,
          fold_res$col_neighbors_list,
          self_include = eval_self_include
        )
    }
    return(list(
      cv_err_mean = mean(cv_results),
      cv_err_sd = sd(cv_results),
      cv_errors = cv_results
    ))
  }


algo_comparison_mse_dataset <- function(mode, mcar_setup = NULL, mnar_setup = NULL, seed_collection = 1:10, percentiles = NULL, row_percentiles = NULL) {
  miss_prob <- 0.5
  mnar_deter <- TRUE
  if(grepl(mode, "MCAR", ignore.case = T)){
    N_values <- mcar_setup$N_values
    rho <- mcar_setup$rho
    noise_sigma <- sqrt(mcar_setup$noise_sigma)
    snr <- mcar_setup$snr
    miss_prob <- mcar_setup$miss_prob
  }
  else if(grepl(mode, "MNAR", ignore.case = T)){
    N_values <- mnar_setup$N_values
    rho <- mnar_setup$rho
    noise_sigma <- sqrt(mnar_setup$noise_sigma)
    snr <- mnar_setup$snr
    mnar_deter <- mnar_setup$mnar_deter
  }

  # percentiles <- c(1, 1.5, seq(2, 8, by = 1), 10)
  if(is.null(percentiles)) {
    # percentiles <- c(seq(1, 2.5, by = 0.5), seq(3, 5, by = 1), seq(6, 12, by = 0.5), 15, 30, 40, 50)
    percentiles <- c(seq(1.5, 5, 0.25), seq(5.5, 8, 0.5), 9, 10)
  }
  if(is.null(row_percentiles)) {
    row_percentiles <- percentiles
  }

  one.sided.percentiles <- c(seq(2, 8, 0.25), seq(8.5, 15, 0.5), seq(16, 30, 1) )

  results <- data.frame(
    N = integer(),
    Algorithm = character(),
    MSE = numeric(),
    stringsAsFactors = FALSE
  )

  for(N in N_values){
    for(seed in seed_collection){
      syn_data <- synthetic_data_gen(N = N, Ts = N, rho = rho, sig_noise = noise_sigma, seed = seed, snr = snr, mode = mode, miss_prob, mnar_deter)
      data_mat <- syn_data$observed_data
      true_data <- syn_data$full_data_true
      noise_sigma <- syn_data$noise_sd
      ans_timenn <-
        cross_val(dataset = data_mat,
                  true_dataset = true_data,
                  func = time_nearest_neighbour_parallel,
                  percentiles = one.sided.percentiles,
                  eval_func = time_nearest_neighbour_evaluator,
                  eval_self_include = F,
                  two_sided_nn = FALSE,
                  verbose = T)

      ans_tsnn <-
        cross_val(dataset = data_mat,
                  true_dataset = true_data,
                  func = double_nearest_neighbour_parallel,
                  percentiles = percentiles,
                  eval_func = double_nearest_neighbour_evaluator,
                  row_percentiles = row_percentiles,
                  eval_self_include = F,
                  two_sided_nn = TRUE,
                  verbose = T)

      ans_drnn <-
        cross_val(dataset = data_mat,
                  true_dataset = true_data,
                  func = doubly_robust_nearest_neighbour_parallel,
                  percentiles = percentiles,
                  eval_func = doubly_robust_nearest_neighbour_evaluator,
                  row_percentiles = row_percentiles,
                  eval_self_include = F,
                  two_sided_nn = TRUE,
                  verbose = T)
      ans_oracle <-
        cross_val_tsnn_oracle(dataset = data_mat,
                              true_dataset = true_data,
                              percentiles = percentiles,
                              row_percentiles = row_percentiles,
                              row_latent = syn_data$row_latent,
                              col_latent = syn_data$col_latent,
                              verbose = T)
      ans_impute <- soft_impute_eval(data_mat, true_data, type = "als")
      ans_usvt <- USVT_mc(true_data, data_mat)

      results <- rbind(results,
                       data.frame(N = N, Algorithm = "TS-NN", MSE = ans_tsnn$cv_err_mean),
                       data.frame(N = N, Algorithm = "DR-NN", MSE = ans_drnn$cv_err_mean),
                       data.frame(N = N, Algorithm = "Oracle TS-NN", MSE = ans_oracle$cv_err_mean),
                       data.frame(N = N, Algorithm = "Soft-Impute", MSE = ans_impute$mse),
                       data.frame(N = N, Algorithm = "USVT", MSE = ans_usvt),
                       # data.frame(N = N, Algorithm = "User-NN", MSE = ans_usernn$cv_err_mean))
                       data.frame(N = N, Algorithm = "Col-NN", MSE = ans_timenn$cv_err_mean))
    }
  }
  # summary_results <- results

  summary_results <- aggregate(
    MSE ~ N + Algorithm,
    data = results,
    FUN = function(x)
      c(mean = mean(x), std = sd(x))
  )

  # Splitting the mean and std into separate columns
  summary_results$Mean_MSE <- summary_results[, 3][, 1]
  summary_results$Std_MSE <- summary_results[, 3][, 2]
  summary_results$MSE <- NULL


  return(list(results = summary_results))
}



mse_comparison_plotter <- function(summary_results, rho, mode) {
  slopes <- summary_results %>%
    dplyr::group_by(Algorithm) %>%
    do({
      model <-
        lm(log(Mean_MSE) ~ log(as.numeric(as.character(N))), data = .)
      print(tidy(model))  # Print intermediate tidy output for debugging
      tidy(model)
    }) %>%
    filter(term == "log(as.numeric(as.character(N)))") %>%
    dplyr::select(Algorithm, slope = estimate)

  summary_results <- summary_results %>%
    left_join(slopes, by = "Algorithm")


  # Create custom legend labels with slopes
  summary_results <- summary_results %>%
    mutate(Algorithm_label = paste(Algorithm, "(Decay: n^", round(slope, 3), ")"))


  #summary_results = summary_results[,-(5:9)]

  # Adjusted custom function to format y-axis labels as 10^a where 'a' can be non-integer
  format_y_as_power <- function(y) {
    # Calculate 'a' as the log base 10 of y, allowing for non-integer values
    a <- log10(y)
    labels <- paste("10^{", sprintf("%.2f", a), "}")
    parse(text = labels)
  }


  # Ensure N is treated as a factor for distinct line colors
  # summary_results$N <- as.numeric(summary_results$N)



  # Create the string manually
  # summary_results$Algorithm_label <- factor(summary_results$Algorithm_label, levels = c("USVT (Decay: n^ 0.044 )", "Soft-Impute (Decay: n^ -0.186 )", "User-NN (Decay: n^ -0.77 )", "DR-NN (Decay: n^ -0.747 )", "TS-NN (Decay: n^ -0.977 )", "Oracle TS-NN (Decay: n^ -1.117 )"))
  # summary_results$Algorithm <- factor(summary_results$Algorithm, levels = c("USVT", "Soft-Impute", "User-NN", "DR-NN", "TS-NN", "Oracle TS-NN"))

  plot_title <- bquote("MSE for (" * lambda * ", 2) holder function in"~.(mode)~ "when " * lambda == .(rho))

  g <- ggplot(
    summary_results,
    aes(
      x = N,
      y = Mean_MSE,
      color = Algorithm_label,
      group = Algorithm_label
      # shape = Algorithm_label,
      # linetype = Algorithm_label
    )
  ) +
    geom_point(size = 5, alpha = 0.7) +
    # geom_line(size = 1) +
    # geom_point(size = 3) +
    # geom_point(size = 4, shape = 17) +  # Points with different shapes for lambda
    # geom_line(aes(linetype = Algorithm), linewidth = 1) +
    geom_smooth(method = "lm",
                se = FALSE, alpha = 0.4) +  # Linear regression lines for each lambda
    geom_errorbar(aes(ymin = Mean_MSE - 2 * Std_MSE, ymax = Mean_MSE + 2 * Std_MSE), width = 3, alpha = 0.7) +
    scale_y_continuous(labels = format_y_as_power, trans = 'log10') +  # Log scale with custom labels
    # scale_linetype_manual(values = c("Oracle TS-NN" = "solid", "TS-NN" = "solid", "DR-NN" = "dotted", "Soft-Impute" = "dotted", "User-NN" = "dotted", "USVT" = "dotted"),
    #                       labels = c("Oracle TS-NN" = "**Oracle TS-NN**", "TS-NN" = "**TS-NN**", "DR-NN" = "DR-NN", "Soft-Impute" = "Soft-Impute", "User-NN" = "User-NN", "USVT" = "USVT")) +
    # scale_linewidth_manual(values = c("Oracle TS-NN" = 3, "TS-NN" = 3, "DR-NN" = 1.5, "Soft-Impute" = 1.5, "User-NN" = 1.5, "USVT" = 1.5),
    #                        labels = c("Oracle TS-NN" = "**Oracle TS-NN**", "TS-NN" = "**TS-NN**", "DR-NN" = "DR-NN", "Soft-Impute" = "Soft-Impute", "User-NN" = "User-NN", "USVT" = "USVT")) +
    # scale_color_manual(
    #   values = c("Oracle TS-NN" = "black", "TS-NN" = "blue", "DR-NN" = "darkgreen", "Soft-Impute" = "orange", "User-NN" = "#FF9999", "USVT" = "darkgray"),
    #   labels = c("Oracle TS-NN" = "**Oracle TS-NN**", "TS-NN" = "**TS-NN**", "DR-NN" = "DR-NN", "Soft-Impute" = "Soft-Impute", "User-NN" = "User-NN", "USVT" = "USVT")
    # ) +
    # geom_point(data = subset(summary_results, Algorithm == "Oracle TS-NN"), shape = 8, size = 4) +
    # geom_point(data = subset(summary_results, Algorithm == "TS-NN"), shape = 9, size = 4) +
    # scale_shape_manual(values = c("Oracle TS-NN" = "star", "TS-NN" = "triangle", "DR-NN" = "square", "Soft-Impute" = "diamond plus", "User-NN" = "diamond", "USVT" = "circle"),
  #                    labels = c("Oracle TS-NN" = "**Oracle TS-NN**", "TS-NN" = "**TS-NN**", "DR-NN" = "DR-NN", "Soft-Impute" = "Soft-Impute", "User-NN" = "User-NN", "USVT" = "USVT")) +
  # scale_color_manual(values = rainbow(length(
  #   unique(summary_results$Algorithm_label)
  # ))) +
  labs(x = "Number of rows, n(= Number of columns, m)",
       y = "Mean Squared Error",
       title = plot_title) +
    theme_bw() +
    # theme_minimal() +
    theme(
      legend.position = c(0.2, 0.2),
      legend.background = element_rect(fill = "transparent", color = NA),
      # strip.text.x = element_blank(),
      plot.title = element_text(face = "bold", size = 22, hjust = 0.5),
      axis.text.x = element_text(size = 20),
      axis.text.y = element_text(size = 20),
      axis.text = element_text(size = 20),
      legend.text = element_text(size = 20),
      legend.title = element_text(size = 20, face = "bold"),
      axis.title = element_text(size = 20, face = "bold"),
      strip.text = element_text(size = 25, face = "bold")
    )
  # +theme(legend.text = element_markdown())
  return(g)
}



mse_sensitivity_lambda_dataset <- function(mode, mcar_setup = NULL, mnar_setup = NULL, seed_collection = 1:10, percentiles = NULL, row_percentiles = NULL) {
  miss_prob <- 0.5
  mnar_deter <- TRUE
  if(grepl(mode, "MCAR", ignore.case = T)){
    N_values <- mcar_setup$N_values
    rho_values <- mcar_setup$rho_values
    noise_sigma <- sqrt(mcar_setup$noise_sigma)
    snr <- mcar_setup$snr
    miss_prob <- mcar_setup$miss_prob
  }else if(grepl(mode, "MNAR", ignore.case = T)){
    N_values <- mnar_setup$N_values
    rho_values <- mnar_setup$rho_values
    noise_sigma <- sqrt(mnar_setup$noise_sigma)
    snr <- mnar_setup$snr
    mnar_deter <- mnar_setup$mnar_deter
  }

  if(is.null(percentiles)) {
    # percentiles <- c(seq(1, 2.5, by = 0.5), seq(3, 5, by = 1), seq(6, 12, by = 0.5), 15, 30, 40, 50)
    # percentiles <- c(seq(2, 5, 0.25), seq(5.5, 8, 0.5), 9, 10, 15)
    percentiles <- c(1, 1.5, 2, seq(3, 8, 1), 10, 15, 30)
  }
  if(is.null(row_percentiles)) {
    row_percentiles <- percentiles
  }

  results <- data.frame(
    lambda = factor(),
    N = integer(),
    MSE = numeric(),
    stringsAsFactors = FALSE
  )
  for(rho in rho_values){
    for(N in N_values){
      for(seed in seed_collection){
        syn_data <- synthetic_data_gen(N = N, Ts = N, rho = rho, sig_noise = noise_sigma, seed = seed, snr = snr, mode = mode, miss_prob, mnar_deter)
        data_mat <- syn_data$observed_data
        true_data <- syn_data$full_data_true
        noise_sigma <- syn_data$noise_sd

        ans_tsnn <-
          cross_val(dataset = data_mat,
                    true_dataset = true_data,
                    func = double_nearest_neighbour_parallel,
                    percentiles = percentiles,
                    row_percentiles = row_percentiles,
                    eval_func = double_nearest_neighbour_evaluator,
                    two_sided_nn = TRUE,
                    verbose = F)

        # ans_impute <- soft_impute_eval(data_mat, true_data, type = "als")
        # ans_usvt <- USVT_mc(true_data, data_mat)

        results <- rbind(results,
                         data.frame(lambda = rho, N = N, MSE = ans_tsnn$cv_err_mean))
      }
    }
  }
  # summary_results <- results

  summary_results <- aggregate(
    MSE ~ N + lambda,
    data = results,
    FUN = function(x)
      c(mean = mean(x), std = sd(x))
  )

  # Splitting the mean and std into separate columns
  summary_results$Mean_MSE <- summary_results[, 3][, 1]
  summary_results$Std_MSE <- summary_results[, 3][, 2]
  summary_results$MSE <- NULL


  return(list(results = summary_results))
}

lambda_sensitivity_plotter <- function(summary_results) {
  slopes <- summary_results %>%
    group_by(lambda) %>%
    do({
      model <-
        lm(log(Mean_MSE) ~ log(as.numeric(as.character(N))), data = .)
      print(tidy(model))  # Print intermediate tidy output for debugging
      tidy(model)
    }) %>%
    filter(term == "log(as.numeric(as.character(N)))") %>%
    select(lambda, slope = estimate)

  print("Slopes calculated for each Algorithm:")
  print(slopes)

  summary_results <- summary_results %>%
    left_join(slopes, by = "lambda")

  # Create custom legend labels with slopes
  summary_results <- summary_results %>%
    mutate(Lambda_label = paste(lambda, "(Decay: n^", round(slope, 3), ")"))



  # summary_results = summary_results[,-(5:9)]

  # Adjusted custom function to format y-axis labels as 10^a where 'a' can be non-integer
  format_y_as_power <- function(y) {
    # Calculate 'a' as the log base 10 of y, allowing for non-integer values
    a <- log10(y)
    labels <- paste("10^{", sprintf("%.2f", a), "}")
    parse(text = labels)
  }


  # Ensure N is treated as a factor for distinct line colors
  summary_results$lambda <- as.factor(summary_results$lambda)


  ggplot(
    summary_results,
    aes(
      x = N,
      y = Mean_MSE,
      color = Lambda_label,
      group = Lambda_label,
      shape = Lambda_label
    )
  ) +
    geom_point(size = 10) +  # Points with different shapes for lambda
    geom_smooth(method = "lm",
                se = FALSE,
                linewidth = 1) +  # Linear regression lines for each lambda
    geom_errorbar(aes(ymin = Mean_MSE - 2 * Std_MSE, ymax = Mean_MSE + 2 * Std_MSE), width = 10) +  # Error bars
    scale_y_continuous(labels = format_y_as_power, trans = 'log10') +  # Log scale with custom labels
    scale_color_manual(values = rainbow(length(unique(
      summary_results$Lambda_label
    )))) +  # Custom colors for lambda
    labs(
      x = "n(=m)",
      y = "MSE",
      title = expression(paste("MSE vs n for TS-NN with Different ", lambda)),
      color = expression(paste(lambda, " value")),
      shape = expression(paste(lambda, " value"))
    ) +
    theme_bw() +
    # theme_minimal() +
    theme(
      legend.position = c(0.2, 0.2),
      legend.background = element_rect(fill = "transparent", color = NA),
      # strip.text.x = element_blank(),
      plot.title = element_text(face = "bold", size = 50, hjust = 0.5),
      axis.text.x = element_text(
        size = 25
      ),
      axis.text.y = element_text(size = 25),
      axis.text = element_text(size = 20),
      legend.text = element_text(size = 28),
      legend.title = element_text(size = 28, face = "bold"),
      axis.title = element_text(size = 50, face = "bold"),
      strip.text = element_text(size = 25, face = "bold")
    )
}

