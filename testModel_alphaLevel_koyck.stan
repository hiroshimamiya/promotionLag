// Stand code for 1st order transfer function to capture Koyck lag, with intercept
// Hiroshi Mamiya 
// July 16, 2021

data {
  int<lower=0> T;   
  real y[T];        
  real x[T];   
}

parameters {
  real <lower=0> sigma_Y; // Standard Deviation (SD) for observation             
  
  vector[T] alpha; 
  real <lower=0> sigma_alpha;  // SD of randomly evolving intercept
   
  real beta; // Immediate effect of explanatory variable
  real <lower = 0, upper = 1> lambda; // Koyck lag coefficient, needs to be -1 < lamgda < 1 
}

transformed parameters { // system equation is specified here
  vector[T] E;
  E[1] = beta*x[1]; 
  for (t in 2:T){ // transfer function of X1 and display 
    E[t] = lambda * E[t-1] + beta*x[t];
  }
}

model { //observation euqation and priors 
  //priors and initial values 
  sigma_Y  ~ cauchy(0, 3); 
  alpha[1] ~ normal(10, 3); 
  sigma_alpha ~ normal(0, 0.5);  
  beta ~ normal(0, 3);
  lambda ~ uniform(0, 1);
  
  // state vector of the level intecept
  for(t in 2:T)alpha[t] ~ normal(alpha[t-1], sigma_alpha);
  //alpha[2:T] ~ normal(alpha[1:(T - 1)], sigma_alpha);//this is faster
  
  // Model the outcome from intercept, lag structure, and noise
  for(t in 1:T) y[t] ~ normal(alpha[t] + E[t], sigma_Y);
  //y ~ normal(alpha + E, sigma_Y);//this is faster
}

// generate predictive distribution and log lik 
generated quantities {
 real yHat[T]; 
 real log_lik [T]; 

 for (t in 1:T){
  yHat[t] = normal_rng(alpha[t] + E[t], sigma_Y);
  log_lik[t] = normal_lpdf(y[t] | alpha[t] + E[t], sigma_Y);
 }
}

