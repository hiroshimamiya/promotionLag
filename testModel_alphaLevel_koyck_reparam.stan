// Stan code to capture Koyck lag, with intercept
// State vector for level is now in transformed parameters section
// Hiroshi Mamiya 
// July 16, 2021


data {
  int<lower=0> T;   
  real y[T];        
  real x[T];   
}

parameters {
  real <lower=0> sigma_V; // Standard Deviation (SD) for observation             
  
  vector[T] sigma_alpha;  // SD of randomly evolving intercept
   
  real beta; // Immediate effect of explanatory variable
  real <lower = 0, upper = 1> lambda; // Koyck lag coefficient, needs to be -1 < lamgda < 1 for stationarity , or 0<lamnda<1 to be monotonic decay
  
  real <lower = 0> scale_alpha; // scaling param for noise of intercept
}

 transformed parameters { // system equation is specified here
  vector[T] E;
  vector[T] alpha; 
  
  E[1] = beta*x[1]; 
   for (t in 2:T)E[t] = lambda * E[t-1] + beta*x[t];
  
  alpha[1] = sigma_alpha[1] ; 
  for(t in 2:T) alpha[t] = alpha[t-1] + sigma_alpha[t]*scale_alpha; 
}

model { //observation euqation and priors 
  // priors 
  sigma_V  ~ normal(0, 5); 
  sigma_alpha ~ normal(0, 1);
  scale_alpha ~ normal(0, 1);
  beta ~ normal(0, 5);
  lambda ~ uniform(0, 1);
  
  // Model the outcome from intercept, lag structure, and noise
  y ~ normal(alpha + E, sigma_V);
}

// generate predictive distribution and log lik 
generated quantities {
 real yHat[T]; 
 real log_lik [T]; 

 for (t in 1:T){
   yHat[t] = normal_rng(alpha[t] + E[t], sigma_V);
  log_lik[t] = normal_lpdf(y[t] | alpha[t] + E[t], sigma_V);
 }
}

