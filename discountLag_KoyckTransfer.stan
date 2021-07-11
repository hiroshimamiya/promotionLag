//  Stan code to run distributed lag model under a transfer function for Koyck specification, integrated into the framework of dynamic learn model 
// The model contain seasonal terms specified by a harmonic wave, covarite, and dynamic intercept with random walk
//Hiroshi Mamiya, McGill University, April 1, 2021


data {
    int<lower=0> T;  // Number of observations (weeks)
    real y[T];    // Outcome, log sales of sugar sweetened beverages     
    real week[T]; // Week index 
    real cov1[T]; //covariate e.g. display promotion 
    real discount[T];   // Mean centered discounting 
    real lLambda; // Lower constraint for lag parameter in transfer function 
    real uLambda; // Upper constraint for lag parameter in transfer function 
    real meanDiscount; // Mean value of discounting, pre-calculated at data preperation and passed to this script 
}

transformed data{ 
  // This section defines season wave 
  vector[T] week_times_pi_freq;
  for (i in 1:T) week_times_pi_freq[i] = 2*week[i]*pi()/52.4; 
}

parameters {
  real <lower=0>  sigma_V;      // Variance of sales 
  vector[T]       alpha;        // Time-varying intercept for stochastic level
  real <lower=0>  sigma_alpha;  // Standard deviation of the intercept
 
  real psi;   // Immediate effect of discounting, *not* time-varying effect 
  real <lower = lLambda, upper = uLambda> lambda; // Upper and lower bound of the lag effect 

  // Covariate (display promotion) and coefficents for season wave
  real betaCov1;
  real betaSin;
  real betaCos;
}


transformed parameters { // system equation is specified here
  vector[T] wave;   // Seaseon wave
  vector[T] E;      // States for transfer function 
  
  // Initialization of state 
  E[1] = psi*discount[1]; 

  // Transfer function for lag - note that there is no noise i.e. deterministic evolution 
  for (t in 2:T){ 
    E[t] = lambda * E[t-1] + psi*discount[t];
  }

  // Linear combination for season wave 
  for (i in 1:T){
    wave[i] = betaSin*sin(week_times_pi_freq[i]) + betaCos*cos(week_times_pi_freq[i]);
  }
}

model { 
  // Observation euqation and priors, also tested with varaince 3^2 and 10^2
  sigma_V  ~ cauchy(0, 5); // also tested for normal prior 
  betaCov1 ~ normal(0, 5);
  betaSin  ~ normal(0, 5);
  betaCos  ~ normal(0, 5); 

  // Prior for intercept 
  alpha[1] ~ normal(0,5);
  sigma_alpha ~ normal(0,1);  

  // Prior for transder function parameters 
  psi ~ normal(0,5);
  lambda ~ uniform(lLambda,uLambda);

  // Model 
  for (t in 1:T){
    y[t] ~ normal(alpha[t] + betaCov1*cov1[t] + wave[t] + E[t], sigma_V);
  }
  for(t in 2:T){
    alpha[t] ~ normal(alpha[t-1], sigma_alpha);
  }
}

// Calculation of log likelihood, predictive distrition, and 
// counterfactual sales quantity comparing fitted mean sales between model with and without lag effects 
// Note that monitoring and saving these variables can make the fit object very large, so unnecessary variables 
// can be ommited. 
generated quantities {
  vector[T] log_lik; 
  vector[T] yHat;
  vector[T] YFit;
  vector[T] YFit_noLag; 
  vector[T] diffY; 
  vector[T] EFit; 

  // Log likelihood and predicted distribution of sales 
  for (t in 1:T){
    yHat[t] = normal_rng(alpha[t] + betaCov1*cov1[t] + wave[t] + E[t], sigma_V); 
    log_lik[t] = normal_lpdf(y[t] | alpha[t] + betaCov1*cov1[t] + wave[t] + E[t], sigma_V);
  }
 
  // Fitted values and difference for time = 1
  EFit[1] = psi*(discount[1] + meanDiscount);
  YFit[1]       = alpha[1] + betaCov1*cov1[1] + wave[1] + EFit[1];
  YFit_noLag[1] = alpha[1] + betaCov1*cov1[1] + wave[1] + EFit[1];
  diffY[1] = exp(YFit[1]) - exp(YFit_noLag[1]);
  
  // Fitted values and difference
  for(i in 2:T){
    EFit[i] = lambda * EFit[i-1] + psi*(discount[i] + meanDiscount);
    YFit[i]       = alpha[i] + betaCov1*cov1[i] + wave[i] + EFit[i];
    YFit_noLag[i] = alpha[i] + betaCov1*cov1[i] + wave[i] + psi*(discount[i] + meanDiscount);
    diffY[i] = exp(YFit[i]) - exp(YFit_noLag[i]);
 }
}

