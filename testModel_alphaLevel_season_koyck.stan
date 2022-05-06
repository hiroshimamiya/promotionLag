// Stan code for 1st order transfer function to capture Koyck lag, with intercept

data {
  int<lower=0> T; // Number of time periods 
  real y[T];  // Outcome vector      
  real x[T];  // Exposure vector
  real weekIndex[T]; // Week index vector  
  real cov1[T];
}


transformed data{ 
  // This section defines season wave 
  vector[T] week_times_pi_freq;
  for (i in 1:T) week_times_pi_freq[i] = 2*weekIndex[i]*pi()/52.4; 
}

parameters {
  real <lower=0> sigma_Y; // Standard Deviation (SD) for observation             
  
  vector[T] alpha; 
  real <lower=0> sigma_alpha;  // SD of randomly evolving intercept
   
  real beta; // Immediate effect of explanatory variable
  real <lower = 0, upper = 1> lambda; // Koyck lag coefficient, needs to be -1 < lamgda < 1 
  
  // Season effects 
  real gammaSin;
  real gammaCos;
  
  // Covariate coefficient
  real deltaCov; 
}

transformed parameters { 
  vector[T] wave;   // Seaseon wave
  vector[T] E;      // Structural varaible of transfer function 
  
  // Initialization
  E[1] = beta*x[1]; 
  
  for (t in 2:T){ // Evolution of transfer function (note that error term was not added to this evolution, but it can be added) 
    E[t] = lambda * E[t-1] + beta*x[t];
  }
  
  // Linear combination of season wave 
  for (i in 1:T){
    wave[i] = gammaSin*sin(week_times_pi_freq[i]) + gammaCos*cos(week_times_pi_freq[i]);
  }
}

model { //Priors, outcome model, and evolution of level
  //priors and initial values 
  sigma_Y  ~ cauchy(0, 5); 
  alpha[1] ~ normal(0, 5); 
  sigma_alpha ~ normal(0, 1);  
  beta ~ normal(0, 5);
  lambda ~ uniform(0, 1);
  gammaSin  ~ normal(0, 5);
  gammaCos  ~ normal(0, 5); 
  deltaCov  ~ normal(0, 5); 
  
  
  // Evoluation of  the level intecept
  //for(t in 2:T)alpha[t] ~ normal(alpha[t-1], sigma_alpha);
  alpha[2:T] ~ normal(alpha[1:(T - 1)], sigma_alpha);//this is faster
  
  // the outcome from intercept, lag structure, season and noise
  for(t in 1:T) y[t] ~ normal(alpha[t] + E[t] + wave[t] + deltaCov*cov1[t], sigma_Y);
}

// generate predictive distribution and log likelihood 
generated quantities {
 real yHat[T]; 
 real log_lik [T]; 

 for (t in 1:T){
  yHat[t] = normal_rng(alpha[t] + E[t] + wave[t] + deltaCov*cov1[t], sigma_Y);
  log_lik[t] = normal_lpdf(y[t] | alpha[t] + E[t] + wave[t] + deltaCov*cov1[t], sigma_Y);
 }
}

