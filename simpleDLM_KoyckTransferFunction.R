### Simulation for koyck lag model with stochastic level (shifting intercept with random walk) -----------------
# Hiroshi Mamiya 
# July/2021

require(rstan)

rm(list=ls())
set.seed(2)

# Numer of observations (time points)
Tn <- 300

# Geometric lag coefficient
lambda <- 0.5 

# Immediate effect 
psi = 0.3

# Observation noise, standard deviation of error for Y 
sigma_Y <- 1

# Standard deviation of shift in intercept 
sigma_alphaLevel <- 0.1



# Intercept
# Initial value, about exp(11) servings of soda typically sold in medium sized supermarket per week
alpha <- numeric(Tn)
# value at time 1
alpha[1] <- rnorm(1, 11, sigma_alphaLevel)

# Generate state vector for time-varying intercept, make sure the values non-negative 
for(t in 2:Tn){
  alpha[t] <- alpha[t-1] + rnorm(1, 0, sigma_alphaLevel)
}


# Explnatory variable that have lagged association , centered
x <- rnorm(Tn,0,1);


# Other variables 
# coefficient for explanatory varaible in Under Koyck lag structure
beta = psi/(1-lambda)
# Observed outcome, log sales quantity
y <- rep(NA,T);
# Mean component of outcome variable consisting of unobserved state vector 
mu <- rep(NA,T);
# Difference between observation and state i.e. [y - mu] for each time 
epsilon <- rep(NA,T); 

# Initialize 
epsilon[1] <- 0
mu[1] <-  alpha[1] + beta*(1-lambda)*x[1]
y[1] <- rnorm(1,mu[1],sigma_Y)

# Generate observations
for (t in 2:Tn){
  # Mean
  mu[t] <- lambda*y[t-1] + (1-lambda)*alpha[t] + beta*(1-lambda)*x[t] - lambda*epsilon[t-1] 
  # Observation, mean plus noise 
  y[t] <- rnorm(1, mu[t], sigma_Y)
  # Noise at time t 
  epsilon[t] <- y[t] - mu[t]
  }


### Compile and execute a Stan model 
T <- Tn
# fit the model 
fit <- stan(file="stan/model/simulation/model_alphaLevel_koyck.stan", data=c("T","y","x"), 
            iter=30000,  chains=3, control = list(max_treedepth = 15))

 


### Inspect the sign of convergence 
traceplot(fit, pars = c("sigma_V", "sigma_alpha", "psi", "lambda", "alpha[1]", "alpha[200]", "alpha[300]"))
ggsave("trace.png", width = 7, height = 4)
# MCMC pairs 
pairs(fit, pars = c("sigma_V", "psi", "lambda", "sigma_alpha", "alpha[1]", "alpha[10]"))



### Results generated below are relevant only if sign of mixing MCMC and convergence is reasonably assumed from traceplot and other means of convergent diagnostics
# Table of posterior summary 
print(fit, pars = c("sigma_V", "psi", "lambda", "sigma_alpha", "alpha[1]", "alpha[10]"))


### Plot posterior distribtion and compare with true value 
plotHist <- function(fit, param, trueValue, range = NA, label = ""){
  # Extract posteiror simulation 
  post <- rstan::extract(fit, pars =  c(eval(param)), permuted = TRUE, inc_warmup = FALSE)
  sim <- post[[1]]
  
  # Plot 
  p <- ggplot() + aes(sim)+ 
    geom_histogram(fill="grey", color="white") + 
    theme_classic() +  
    geom_vline(xintercept = trueValue, lty = 5) + 
    xlab(label) + ylab("") +
    theme(axis.text = element_text(size = 12))
  
  # Range of X axist if needed
  if(!is.na(range)){
    p <- p + scale_x_continuous(limits=c(range[1],range[2]))  
  }
  return(p)
}

plotList <- list()
plotList[[1]] <- plotHist(fit,  param = "lambda", trueValue = lambda, range = c(0,1), label = expression(lambda))
plotList[[2]] <- plotHist(fit,  param = "psi", trueValue = psi, label = expression(psi))
plotList[[3]] <- plotHist(fit,  param = "sigma_alpha", trueValue = sigma_alphaLevel, label = expression(sigma[alpha]))
plotList[[4]] <- plotHist(fit,  param = "sigma_V", trueValue = sigma_Y, label = expression(sigma[Y]))

cowplot::plot_grid(plotlist = plotList, ncol = 2, labels = "auto", label_fontface = "plain")

ggsave("sim_postParams.png", width = 7, height = 6)




### Generate Impuse response function
post <- rstan::extract(fit, pars =  c("psi", "lambda"), permuted = TRUE, inc_warmup = FALSE)
psiPosterior = post[[1]]; lambdaPosterior = post[[2]]
h <- 8
# IRF, mean, lower and uppfer CI 
irfWeight <- data.frame(weekLag = 1:h, lo = rep(NA, h), median = rep(NA, h),  hi = rep(NA, h))
# Time t
irfWeight[1, ] <- c(1, quantile(psiPosterior , c(0.025)), median(psiPosterior), quantile(psiPosterior , c(0.975)))
# time t+(2:h)
for(i in 2:h){irfWeight[i, ] <- c(i, quantile(psi * {lambdaPosterior^(i-1)}, c(0.025, 0.5,0.975)))} 

# Plot Impulse response function 
irfWeight <-  irfWeight %>% 
  mutate(lo = lo, hi = hi, median = median, weekLag = weekLag - 1)
scaleFUN <- function(x) sprintf("%.1f", x)
 ggplot(data = irfWeight, aes(x=(weekLag), y=median)) + 
  geom_line() + 
  theme_classic() +
  xlab("Lag") + ylab("Change of outcome") + 
  geom_ribbon(aes(ymin=lo, ymax=hi), linetype=2, alpha=0.1)  + 
  scale_y_continuous(labels=scaleFUN) + 
  theme(axis.text = element_text(color = "black"))
ggsave("sim_IRF.png", width = 5, height = 3)


# Compare fitted and unobserved (true) intercept 
params <- rstan::extract(fit, pars = "alpha")
mean <- apply(params$alpha, 2, mean)
lo <- apply(params$alpha, 2, quantile, 0.025)
hi <- apply(params$alpha, 2, quantile, 0.975)
png("sim_alphaFit.png", res = 300, units = "in", width = 7, height = 4)
plot(lo, type = "l", lty = "dotted", ylim = range(c(mean, lo, hi)), ylab = expression(alpha[t]), xlab = "Time")
points(alpha, pch = 3, cex = 0.5, col = "blue")
lines(hi, lty = "dotted")
lines(mean, type = "l", lwd = 1)
dev.off()







