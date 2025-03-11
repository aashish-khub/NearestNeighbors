library(foreach)
library(doParallel)
library(matrixStats)
library(Rcpp)
library(dplyr)
library(ggplot2)
library(scales)
library(broom)
library(stringr)

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

  eta1_values <- quantile(row_distances[row_distances != Inf], row_percentiles / 100)
  eta2_values <- quantile(col_distances[col_distances != Inf], percentiles / 100)
  nloop <- length(eta1_values)

  # Register parallel backend
  num_cores <- detectCores()

  results <- mclapply(1:nloop, function(index) {
    intermediate_results <- NULL
    eta1 <- eta1_values[index]
    index_set_of_eta2 <- (max(1, index-4): min(index+4, nloop))
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
  }, mc.cores = num_cores - 2)

  results <- do.call(rbind, results)
  results
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
      "Minimum MSE:", best_mse,
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
  function(valid_data, obs_data, row_neighbors_list, col_neighbors_list, verbose = TRUE, self_include = TRUE) {

    predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
    for (u in 1:nrow(obs_data)) {
      for (i in 1:ncol(obs_data)) {
        row_neighbours <- row_neighbors_list[[u]]
        col_neighbours <- col_neighbors_list[[i]]
        if(!self_include) {
          temp <- obs_data[u, i]
          obs_data[u, i] <- NA
        }
        values <- valid_data[row_neighbours, col_neighbours]
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

    mat <- (valid_data - predicted_X) ^ 2
    eval_errors <- as.vector(mat[!is.na(mat) & !is.nan(mat)])

    # You would need to adjust this part to return the best_predicted_X
    return(sqrt(eval_errors))
  }

unit_nearest_neighbour_parallel <- function(valid_data, obs_data, percentiles = c(10, 13, 17, 20), self_include = FALSE, verbose = TRUE) {
  row_distances <- compute_distances_rcpp(obs_data, TRUE)
  eta1_values <- quantile(row_distances[row_distances != Inf], percentiles / 100)
  n <- nrow(obs_data)
  num_cores <- detectCores()

  results <- mclapply(eta1_values, function(eta1) {
    predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
    for (u in 1:n) {
      row_neighbours <- which(row_distances[u, ] <= eta1)
      if(!self_include) {
        if (u %in% row_neighbours) {
          row_neighbours <- row_neighbours[-which(row_neighbours == u)]
        }
      }
      temp <- obs_data[u, ]
      obs_data[u, ] <- NA
      if (length(row_neighbours) == 1) {
        predicted_X[u, ] <- obs_data[row_neighbours, ]
      }
      else if (length(row_neighbours) > 1) {
        predicted_X[u, ] <- colMeans(obs_data[row_neighbours, ], na.rm = TRUE)
      }
      obs_data[u, ] <- temp
    }

    mse <- mean((valid_data - predicted_X) ^ 2, na.rm = TRUE)

    return(data.frame(eta1 = eta1, mse = mse))
  }, mc.cores = num_cores - 4)

  results <- do.call(rbind, results)

  # Find best result
  best_idx <- which.min(results$mse)
  best_eta1 <- results$eta1[best_idx]
  best_mse <- results$mse[best_idx]

  row_dist_cdf <- ecdf(row_distances[row_distances != Inf])
  eta1_percentile <- row_dist_cdf(best_eta1)

  row_neighbours <- list()
  for(u in 1:n) {
    row_neighbours[[u]] <- which(row_distances[u, ] <= best_eta1)
  }

  if (verbose) {
    cat(
      "Row-NN Minimum MSE:", best_mse,
      "Eta1 percentile:", eta1_percentile, "\n"
    )
  }

  return(list(
    eta1 = best_eta1,
    mse = best_mse,
    eta_percentile = eta1_percentile,
    neighbors_list = row_neighbours
  ))
}

unit_nearest_neighbour_evaluator <-
  function(valid_data, obs_data, neighbors_list, verbose = TRUE, self_include = TRUE, true_signal = NULL) {
    predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
    for (u in 1:nrow(obs_data)) {
      row_neighbours <- neighbors_list[[u]]
      if(!self_include) {
        if (u %in% row_neighbours) {
          row_neighbours <- row_neighbours[-which(row_neighbours == u)]
        }
      }
      temp <- obs_data[u, ]
      obs_data[u, ] <- NA
      if (length(row_neighbours) == 1) {
        predicted_X[u, ] <- obs_data[row_neighbours, ]
      }
      else if (length(row_neighbours) > 1) {
        predicted_X[u, ] <- colMeans(obs_data[row_neighbours,], na.rm = TRUE)
      }
      obs_data[u, ] <- temp
    }

    mat <- (valid_data - predicted_X) ^ 2
    eval_errors <- as.vector(mat[!is.na(mat) & !is.nan(mat)])

    return(sqrt(eval_errors))
  }

time_nearest_neighbour_parallel <- function(valid_data, obs_data, percentiles = c(10, 13, 17, 20), self_include = FALSE, verbose = TRUE) {
  col_distances <- compute_distances_rcpp(obs_data, FALSE)
  eta2_values <- quantile(col_distances[col_distances != Inf], percentiles / 100)

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
      "Minimum MSE:", best_mse,
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
  mat <- (valid_data - predicted_X) ^ 2
  eval_errors <- as.vector(mat[!is.na(mat) & !is.nan(mat)])

  # You would need to adjust this part to return the best_predicted_X
  return(sqrt(eval_errors))
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
    quantile(row_distances[row_distances != Inf], row_percentiles / 100)
  eta2_values <-
    quantile(col_distances[col_distances != Inf], percentiles / 100)
  nloop <- length(eta1_values)

  # Register parallel backend
  num_cores <- detectCores()
  registerDoParallel(cores = num_cores - 1)

  results <-
    foreach(
      index = 1:nloop,
      .combine = rbind,
      .packages = c('matrixStats')
    ) %dopar% {
      eta1 <- eta1_values[index]
      index_set_of_eta2 <- c(max(1, index - 5), min(index + 5, nloop))
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

    mat <- (valid_data - predicted_X) ^ 2
    eval_errors <- as.vector(mat[!is.na(mat) & !is.nan(mat)])

    # You would need to adjust this part to return the best_predicted_X
    return(sqrt(eval_errors))
  }

cross_val_real_life <- function(dataset, func, percentiles, eval_func, nfolds = 5, two_sided_nn = TRUE, verbose = FALSE, self_include = FALSE, row_percentiles = NULL, eval_self_include = FALSE) {
  n <- nrow(dataset)
  m <- ncol(dataset)
  cv_results <- NULL
  test_ncol <- 40
  for(fold in 1:nfolds) {
    test_folds <- (round((fold - 1) * n/nfolds) + 1):round((fold * n/nfolds))
    test_data <- matrix(NA, nrow = n, ncol = m)
    test_data[test_folds, ((m - test_ncol):m)] <- dataset[test_folds, ((m - test_ncol):m)]
    train_data <- dataset
    train_data[test_folds, ((m - test_ncol):m)] <- NA
    if(two_sided_nn){
      fold_res <- func(train_data, train_data, percentiles, verbose = verbose, self_include = self_include, row_percentiles = row_percentiles)
      cv_results <- c(cv_results, eval_func(test_data, train_data, fold_res$row_neighbors_list, fold_res$col_neighbors_list, self_include = eval_self_include))
    }
    else{
      fold_res <- func(train_data, train_data, percentiles, verbose = verbose, self_include = self_include)
      cv_results <- c(cv_results, eval_func(test_data, train_data, fold_res$neighbors_list, self_include = eval_self_include))
    }
  }
  return(list(cv_err_mean = mean(cv_results),
              cv_err_sd = sd(cv_results),
              cv_errors = cv_results))
}

all_unit_nn <- function(valid_data, obs_data, self_include = F, verbose = T) {
  n <- nrow(obs_data)

  predicted_X <- matrix(NA, nrow(obs_data), ncol(obs_data))
  for (u in 1:n) {
    row_neighbours <- 1:n
    if(!self_include) {
      if (u %in% row_neighbours) {
        row_neighbours <- row_neighbours[-which(row_neighbours == u)]
      }
    }
    temp <- obs_data[u, ]
    obs_data[u, ] <- NA
    if (length(row_neighbours) == 1) {
      predicted_X[u, ] <- obs_data[row_neighbours, ]
    }
    else if (length(row_neighbours) > 1) {
      predicted_X[u, ] <- colMeans(obs_data[row_neighbours,], na.rm = TRUE)
    }
    obs_data[u, ] <- temp
  }

  mat <- (valid_data - predicted_X) ^ 2
  eval_errors <- as.vector(mat[!is.na(mat) & !is.nan(mat)])

  # You would need to adjust this part to return the best_predicted_X
  return(sqrt(eval_errors))
}

all_time_nn <- function(valid_data, obs_data, self_include = F, verbose = T) {
  m <- ncol(obs_data)
  predicted_X <- matrix(NA, nrow(obs_data), m)

  for (i in 1:m) {
    col_neighbours <- 1:m
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
  mat <- (valid_data - predicted_X) ^ 2
  eval_errors <- as.vector(mat[!is.na(mat) & !is.nan(mat)])

  # You would need to adjust this part to return the best_predicted_X
  return(sqrt(eval_errors))
}

cross_val_unit_benchmark <- function(dataset, nfolds = 5, verbose = T, self_include = FALSE) {
  n <- nrow(dataset)
  m <- ncol(dataset)
  cv_results <- NULL
  test_ncol <- 40
  for(fold in 1:nfolds) {
    test_folds <- (round((fold - 1) * n/nfolds) + 1):round((fold * n/nfolds))
    test_data <- matrix(NA, nrow = n, ncol = m)
    test_data[test_folds, ((m - test_ncol):m)] <- dataset[test_folds, ((m - test_ncol):m)]
    train_data <- dataset
    train_data[test_folds, ((m - test_ncol):m)] <- NA
    cv_results <- c(cv_results, all_unit_nn(test_data, train_data, self_include = self_include, verbose = verbose))
  }
  return(list(cv_err_mean = mean(cv_results),
              cv_err_sd = sd(cv_results),
              cv_errors = cv_results))
}

cross_val_time_benchmark <- function(dataset, nfolds = 5, verbose = T, self_include = FALSE) {
  n <- nrow(dataset)
  m <- ncol(dataset)
  cv_results <- NULL
  test_ncol <- 40
  for(fold in 1:nfolds) {
    test_folds <- (round((fold - 1) * n/nfolds) + 1):round((fold * n/nfolds))
    test_data <- matrix(NA, nrow = n, ncol = m)
    test_data[test_folds, ((m - test_ncol):m)] <- dataset[test_folds, ((m - test_ncol):m)]
    train_data <- dataset
    train_data[test_folds, ((m - test_ncol):m)] <- NA
    cv_results <- c(cv_results, all_time_nn(test_data, train_data, self_include = self_include, verbose = verbose))
  }
  return(list(cv_err_mean = mean(cv_results),
              cv_err_sd = sd(cv_results),
              cv_errors = cv_results))
}


algo_comparison_mse_real_life <-
  function(data_mat,
           percentiles,
           one.side.nn.percentiles,
           row_percentiles = NULL,
           drnn_row_percentiles = NULL,
           drnn_col_percentiles = NULL) {

    if(is.null(drnn_row_percentiles)) {
      drnn_row_percentiles <- row_percentiles
    }
    if(is.null(drnn_col_percentiles)) {
      drnn_col_percentiles <- percentiles
    }

    results <- NULL

    ans_timenn <-
      cross_val_real_life(dataset = data_mat,
                          func = time_nearest_neighbour_parallel,
                          percentiles = one.side.nn.percentiles,
                          eval_func = time_nearest_neighbour_evaluator,
                          two_sided_nn = FALSE,
                          verbose = T)

    ans_usernn <-
      cross_val_real_life(dataset = data_mat,
                          func = unit_nearest_neighbour_parallel,
                          percentiles = one.side.nn.percentiles,
                          eval_func = unit_nearest_neighbour_evaluator,
                          two_sided_nn = FALSE,
                          verbose = T)

    ans_tsnn <-
      cross_val_real_life(dataset = data_mat,
                          func = double_nearest_neighbour_parallel,
                          percentiles = percentiles,
                          eval_func = double_nearest_neighbour_evaluator,
                          two_sided_nn = TRUE,
                          verbose = T,
                          row_percentiles = row_percentiles)

    ans_drnn <-
      cross_val_real_life(dataset = data_mat,
                          func = doubly_robust_nearest_neighbour_parallel,
                          percentiles = drnn_col_percentiles,
                          eval_func = doubly_robust_nearest_neighbour_evaluator,
                          two_sided_nn = TRUE,
                          verbose = T,
                          row_percentiles = drnn_row_percentiles)

    ans_all_user <-
      cross_val_unit_benchmark(dataset = data_mat)


    ans_all_time <-
      cross_val_time_benchmark(dataset = data_mat)

    for(i in 1:length(ans_tsnn$cv_errors)) {
      results <- rbind(results,
                       data.frame(Algorithm = "TS-NN", MSE = ans_tsnn$cv_errors[i]))
    }

    for(i in 1:length(ans_drnn$cv_errors)) {
      results <- rbind(results,
                       data.frame(Algorithm = "DR-NN", MSE = ans_drnn$cv_errors[i]))
    }

    for(i in 1:length(ans_usernn$cv_errors)) {
      results <- rbind(results,
                       data.frame(Algorithm = "User-NN", MSE = ans_usernn$cv_errors[i]))
    }

    for(i in 1:length(ans_timenn$cv_errors)) {
      results <- rbind(results,
                       data.frame(Algorithm = "Time-NN", MSE = ans_timenn$cv_errors[i]))
    }

    for(i in 1:length(ans_all_time$cv_errors)) {
      results <- rbind(results,
                       data.frame(Algorithm = "All-Time-NN", MSE = ans_all_time$cv_errors[i]))
    }

    for(i in 1:length(ans_all_user$cv_errors)) {
      results <- rbind(results,
                       data.frame(Algorithm = "All-User-NN", MSE = ans_all_user$cv_errors[i]))
    }

    return(results)
  }


#jbsteps 30 notification sent matrix (store the data files in your working directory)
dataset = read.csv("log_jbsteps30_noti_sent.csv")

#jbsteps 30 noti not sent dataset (store the data files in your working directory)
# dataset = read.csv("log_jbsteps30_noti_not_sent.csv")


dataset = as.matrix(dataset[, 1:210])
percentiles = c(seq(8, 13, 1), seq(15, 25, by = 1))
one.side.nn.percentiles = c(seq(25, 90, by = 2.5))
row_percentiles = c(seq(15, 20, 1), seq(21, 25, by = 2))
drnn_row_per = seq(25, 85, 2.5)
drnn_col_per = seq(25, 85, 2.5)

final_results = algo_comparison_mse_real_life(
  data_mat = dataset,
  percentiles = percentiles,
  one.side.nn.percentiles = one.side.nn.percentiles,
  row_percentiles = row_percentiles,
  drnn_row_percentiles = drnn_row_per,
  drnn_col_percentiles = drnn_col_per
)



# Update the Algorithm names
final_results <- final_results %>%
  mutate(Algorithm = str_replace_all(Algorithm, c(
    "User-NN" = "Row-NN",
    "All-User-NN" = "All-Row-NN",
    "Time-NN" = "Col-NN",
    "All-Time-NN" = "All-Col-NN"
  )))

final_results <- final_results %>%
  mutate(Algorithm = str_replace_all(Algorithm, c(
    "User-NN" = "Row-NN",
    "All-Row-NN" = "AllRow-NN",
    "Time-NN" = "Col-NN",
    "All-Col-NN" = "AllCol-NN"
  )))

# View the updated data frame
print(unique(final_results$Algorithm))
custom_colors <-
  c(
    "TS-NN" = "#0072B2",
    "DR-NN" = "#E69F00",
    # "Row-NN" = "green",
    "Row-NN" = "#FF9999",
    "Col-NN" = "#F0E442",
    "AllCol-NN" = "red",
    "AllRow-NN" = "green"
    # "AllRow-NN" = "maroon"
  )


g3 <-
  ggplot(final_results,
         aes(x = Algorithm, y = MSE, fill = Algorithm)) +
  geom_boxplot(width = 0.5, linewidth = 1, outlier.size = 2, alpha = 0.8, outlier.shape = 15) +
  scale_fill_manual(values = custom_colors) +
  labs(x = "Algorithm", y = "Error") +
  theme_bw()+
  theme(
    legend.position = "none",
    # strip.text.x = element_blank(),
    plot.title = element_text(size = 25, hjust = 0.5),
    axis.text.x = element_text(size = 21, angle = 0, hjust = 0.5),
    axis.text.y = element_text(size = 20),
    panel.border = element_blank(),
    axis.line.x.bottom = element_line(linewidth = 1.3, color = "black"),
    axis.line.y.left = element_line(linewidth = 1.3, color = "black"),
    panel.grid.major = element_line(color = "gray80", size = 0.75),
    panel.grid.minor = element_blank(),
    # legend.text = element_text(size = 8, face = "bold"),
    #legend.title=element_text(size = 15, face = "bold"),
    axis.title.x = element_text(size = 30, margin = margin(t = 10)),
    axis.title.y = element_text(size = 30, margin = margin(r = 10)),
    strip.text = element_text(size = 15)
  )
print(g3)