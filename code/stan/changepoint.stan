#include functions.stan

data {
  int<lower=0> T;
  int<lower=0> T_prime;
  int<lower=0> N;
  int<lower=0> M;
  array[T + T_prime] int<lower=1> ii;
  array[T + T_prime] int<lower=1> jj;
  array[T + T_prime] int<lower=0> B;
  vector[T + T_prime] y;
  int<lower=0, upper=1> learn;
  real<lower=0> MAX_PRED;
}

parameters {
  vector[M - 1] alpha_star;
  real<lower=0> omega_star;
  real beta_star;
  vector[2] gamma;
}

transformed parameters {
  vector[M - 1] lp = rep_vector(-log(M - 1), M - 1);
  vector<lower=0>[M - 1] alpha = exp(alpha_star);
  real<lower=0> omega = exp(omega_star);
  real<lower=0, upper=1> beta = inv_logit(beta_star);

  for(tau in 2:M) {
    for(t in 1:T){
      int lag = jj[t];
      int year = ii[t];

      if(lag > 1){
        real mu;
        real sigma2 = exp(gamma[1] + gamma[2] * lag + log(y[B[t]]));
        if(lag < tau){
          mu = alpha_star[lag - 1] + log(y[B[t]]);
          lp[tau - 1] += lognormal_lpdf(y[B[t] + 1] | mu, sqrt(sigma2));
        }
        else {
          mu = omega_star * pow(beta, lag) + log(y[B[t]]);
          lp[tau - 1] += lognormal_lpdf(y[B[t] + 1] | mu, sqrt(sigma2));
        }
      }
    }
  }
}

model {
  for(j in 1:M - 1){
    alpha_star[j] ~ normal(0, 1.0 / j);
  }
  omega_star ~ normal(0, 1);
  beta_star ~ normal(0, 1);
  gamma ~ normal(0, 1);

  if(learn) target += log_sum_exp(lp);
}

generated quantities {
  array[T + T_prime] real<lower=0> y_tilde;
  array[T + T_prime] real log_lik;
  simplex[M - 1] p_tau;
  int<lower=2, upper=M> tau_star;

  p_tau = softmax(lp);
  tau_star = categorical_rng(p_tau) + 1;

  for(t in 1:(T + T_prime)){
    real mu;
    real sigma2;
    int lag = jj[t];
    int year = ii[t];
    real lagged_y;

    if(lag > 1) {
      if(isin(B[t], B[1:T]))
        lagged_y = y[B[t]];
      else 
        lagged_y = y_tilde[B[t]];
    }
    if(lag < tau_star && lag > 1){
      mu = log(lagged_y) + alpha_star[lag - 1];
      sigma2 = exp(gamma[1] + gamma[2] * lag + log(lagged_y));
    }
    else if(lag >= tau_star) {
      mu = log(lagged_y) + omega_star * pow(beta, lag);
      sigma2 = exp(gamma[1] + gamma[2] * lag + log(lagged_y));
    }
    if(lag == 1){
      y_tilde[B[t] + 1] = y[B[t] + 1];
      log_lik[B[t] + 1] = 0.0;
    } else {
      y_tilde[B[t] + 1] = min([MAX_PRED, lognormal_rng(mu, sqrt(sigma2))]);
      log_lik[B[t] + 1] = lognormal_lpdf(y[B[t] + 1] | mu, sqrt(sigma2));
    }
  }
}
