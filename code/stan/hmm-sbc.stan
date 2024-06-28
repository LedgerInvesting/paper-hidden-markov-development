#include functions.stan

data {
  int<lower=0> T;
  int<lower=0> T_prime;
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> K;
  array[T + T_prime] int<lower=1> ii;
  array[T + T_prime] int<lower=1> jj;
  array[T + T_prime] int<lower=0> B;
  vector[T + T_prime] y;
  int<lower=0, upper=1> learn;
}

transformed data {
  int nu = 0;
}

parameters {
  vector[M - 1] alpha_star;
  real<lower=0> omega_star;
  real beta_star;
  vector[K] gamma;
  real pi_star;
}

transformed parameters {
  vector[N] lp = rep_vector(0.0, N);
  matrix[T, K] lp_steps;
  matrix[T, K] lp_paths;
  vector<lower=0>[M - 1] alpha = exp(alpha_star);
  real<lower=0> omega = exp(omega_star);
  real<lower=0, upper=1> beta = inv_logit(beta_star);
  real<lower=0, upper=1> pi_ = inv_logit(pi_star);

  for(t in 1:T){
    vector[K] mu;
    real sigma2;
    matrix[K, K] theta;
    int lag = jj[t];
    int year = ii[t];

    theta[1, 1] = pi_;
    theta[1, 2] = 1 - pi_;
    theta[2, 1] = nu;
    theta[2, 2] = 1 - nu;

    if(lag > 1){
      mu[1] = alpha_star[lag - 1] + log(y[B[t]]);
      mu[2] = omega_star * pow(beta, lag) + log(y[B[t]]);
      sigma2 = exp(gamma[1] + gamma[2] * lag + log(y[B[t]]));
    }

    if(lag == 1){
      lp_steps[t] = [0, negative_infinity()];
      lp_paths[t] = lp_steps[t];
    } else if(lag == 2){
      for(k in 1:K){
          lp_steps[t, k] = lognormal_lpdf(y[B[t] + 1] | mu[k], sqrt(sigma2));
          lp_paths[t, k] = log(theta[1, k]) + lp_steps[t, k];
        }
    } else {
      for(k in 1:K){
          lp_steps[t, k] = lognormal_lpdf(y[B[t] + 1] | mu[k], sqrt(sigma2));
          lp_paths[t, k] = log_sum_exp(
            lp_paths[t - 1, 1] + log(theta[1, k]) + lp_steps[t, k],
            lp_paths[t - 1, 2] + log(theta[2, k]) + lp_steps[t, k]
          );
        }
      }

    if(t == T || year != ii[t + 1]){
      lp[year] = log_sum_exp(lp_paths[t]);
    }
  }
}

model {
  for(j in 1:M - 1){
    alpha_star[j] ~ normal(0, 1.0 / j);
  }
  omega_star ~ normal(0, 1);
  beta_star ~ normal(0, 1);
  gamma[1] ~ normal(-3, 0.25);
  gamma[2] ~ normal(-1, 0.1);
  pi_star ~ normal(0, 1);

  if(learn) target += sum(lp);
}

generated quantities {
  array[T + T_prime] int<lower=1, upper=K> z_star;
  array[T + T_prime] real<lower=0> y_tilde;
  array[T + T_prime] real log_lik;
  matrix[T + T_prime, K] lp_pred_steps;
  array[T + T_prime, K] real best_logp;
  array[T + T_prime, K, K] real logp;
  real log_p_z_star;
  vector[K] mu;
  
  {
    matrix[K, K] theta;
    real lagged_y;
    real sigma2;

    for(t in 1:T + T_prime){
      int lag = jj[t];
      int year = ii[t];

      if(lag > 1){
        theta[1, 1] = pi_;
        theta[1, 2] = 1 - pi_;
        theta[2, 1] = nu;
        theta[2, 2] = 1 - nu;
        if(isin(B[t], B[1:T]))
          lagged_y = y[B[t]];
        else 
          lagged_y = y_tilde[B[t]];
        mu[1] = alpha_star[lag - 1] + log(lagged_y);
        mu[2] = omega_star * pow(beta, lag) + log(lagged_y);
        sigma2 = exp(gamma[1] + gamma[2] * lag + log(lagged_y));
      }

      if(lag == 1){
        best_logp[B[t] + 1] = {0, negative_infinity()};
        lp_pred_steps[B[t] + 1] = [0, 0];
        z_star[B[t] + 1] = 1;
        log_lik[B[t] + 1] = 0.0;
        y_tilde[B[t] + 1] = y[B[t] + 1];
      } else {
          for(k in 1:K){
            best_logp[B[t] + 1, k] = negative_infinity();
            lp_pred_steps[B[t] + 1, k] = lognormal_lpdf(y[B[t] + 1] | mu[k], sqrt(sigma2));
            for(j in 1:K){
              if(lag == 2)
                logp[B[t] + 1, j, k] = log(theta[1, k]) + lp_pred_steps[B[t] + 1, k];
              else
                logp[B[t] + 1, j, k] = best_logp[B[t], j] + log(theta[j, k]) + lp_pred_steps[B[t] + 1, k];
              if(logp[B[t] + 1, j, k] > best_logp[B[t] + 1, k]){
                best_logp[B[t] + 1, k] = logp[B[t] + 1, j, k];
              }
            }
            if(t <= T)
              z_star[B[t] + 1] = best_logp[B[t] + 1, 1] > best_logp[B[t] + 1, 2] ? 1 : 2;
            else
              z_star[B[t] + 1] = categorical_rng(theta[z_star[B[t]]]');
            log_lik[B[t] + 1] = lognormal_lpdf(y[B[t] + 1] | mu[z_star[B[t] + 1]], sqrt(sigma2));
            y_tilde[B[t] + 1] = lognormal_rng(mu[z_star[B[t] + 1]], sqrt(sigma2));
          }
        }
      } 
  }
}
