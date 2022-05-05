
rm(list=ls())

require(rstan)
require(ggplot2)
require(cowplot)

options(mc.cores=parallel::detectCores())

set.seed(101)
# Function to generate outcome from lagged exposure-outcome association
# Function to generate matrix of exposure with lag, borrowed from : https://github.com/alastairrushworth/badlm/blob/master/R/lag_matrix.R
lag_matrix  <-  function (exposure, p, start.at.zero = T){
  windows <- function(n) rev(exposure[n:(n + p - !start.at.zero)])
  nums <- as.list(1:(length(exposure) - p))
  matrix(unlist(lapply(nums, windows)), nrow = (length(exposure) - p), ncol = p + start.at.zero, byrow = TRUE)
}








### sample data ----------------------------------------------------------------
Tn <- 300

# Lag coefficient
lambda <- 0.5
# Immediate effect 
beta = 0.65
# Observation noise, standard deviation of error for Y
sigma_Y <- 1
# Standard deviation of shift in intercept 
sigma_alphaLevel <- 0.2


# Intercept, e.g.  log store-level sales of junk food, alpha_1 (initial value) is log(10)
alpha <- cumsum(rnorm(n = Tn, sd = sigma_alphaLevel)) + 10



H <- 0:20 # Max time periods in lag function 
lagFunction <- beta*lambda^(H) #exponential decay 
#lagFunction <- (beta*exp(-(1-lambda)*H)) # more gradual decay

x <- arima.sim(model = list(ar = 0.6), n = Tn+max(H), sd = 0.1)
x <- scale(x)[,1]


lag_mat       <- lag_matrix(x, p = max(H))

# Generate outcome, combination of lagged effect of x, intercept and noise 
yMean    <- as.numeric(lag_mat %*% lagFunction) + alpha
# Add noise
y <- yMean + rnorm(Tn, mean = 0, sd = sigma_Y)

T <- Tn # Time variable, 
x <- x[-H] # Trim lag horizon

### plot 
plot(lagFunction, type = "l", main = "Lag function", xlab = "Time lag", ylab = "Association")
par(mfrow=c(2,1))
plot(x, main = "Exposure", type = "l", xlab = "time")
plot(y, main = "Outcome", type = "l", xlab = "time")
par(mfrow=c(1,1))




###Run-------------------------------------------------------------------------
# Fitting time-series model, state vector (alpha) written in the model block   
fit3 <- stan(file="testModel_alphaLevel_koyck.stan", data=c("T","y","x"), 
             iter=20000,  chains=3, control = list(max_treedepth = 15))

# Quick check of mcmc diagnosis
traceplot(fit3, pars = c("sigma_Y", "sigma_alpha", "beta", "lambda", "alpha[2]", "alpha[200]", "alpha[300]")) + ggtile("Trace of MCMC")
pairs(fit3, pars = c("sigma_Y", "sigma_alpha",  "sigma_alpha", "beta", "lambda", "alpha[2]", "alpha[200]", "alpha[300]")) + ggtitle("Bivariate distribution of MCMC")
sum(summary(fit3)$summary[,"Rhat"] > 1.01)

# further checks 
source("https://raw.githubusercontent.com/betanalpha/knitr_case_studies/master/stan_intro/stan_utility.R")
check_all_diagnostics(fit3)

# Autocorrelation function 
fittedPosterior <- rstan::extract(fit3, pars = "yHat")
resids <- apply(fittedPosterior$yHat, 2, mean) - y
acf(resids, main = "Autocorrelation function of residuals")
hist(resids, main = "Histogram of residuals")







### Results --------------------------------------------------------------------
# Estimated impulse response function  
plot(fit3, 
     ci_level = 0.95,
     pars = c("beta", "lambda", "sigma_alpha", "sigma_Y")) + 
  ggtitle("Summary of recovered parameters (95% Interval)")


#Posterior of transfer function 
post <- rstan::extract(fit3, pars =  c("beta", "lambda"), permuted = TRUE, inc_warmup = FALSE)
betaPosterior = post[[1]]
lambdaPosterior = post[[2]]


# Summary of recovered parameters 
quantile(betaPosterior, probs = c(0.025, 0.5, 0.975))
quantile(lambdaPosterior, probs = c(0.025, 0.5, 0.975))


# Plot Impulse response function
h <- 8
# IRF, mean, lower and uppfer CI 
irfWeight <- data.frame(weekLag = 1:h, lo = rep(NA, h), mean = rep(NA, h),  hi = rep(NA, h))
# Time t
irfWeight[1, ] <- c(1, quantile(betaPosterior , c(0.025)), mean(betaPosterior), quantile(betaPosterior , c(0.975)))
# time t+(2:h)
for(i in 2:h){irfWeight[i, ] <- c(i, quantile(beta * {lambdaPosterior^(i-1)}, c(0.025, 0.5,0.975)))} 

# Plot Impulse response function 
irfWeight$lagTrue <-  lagFunction[1:h] 

scaleFUN <- function(x) sprintf("%.1f", x)
p <-  ggplot(data = irfWeight, aes(x=(weekLag), y=mean)) + 
  geom_line() + 
  theme_classic() +
  xlab("Lag") + ylab("Change of outcome") + 
  geom_ribbon(aes(ymin=lo, ymax=hi), linetype=2, alpha=0.1)  + 
  scale_y_continuous(labels=scaleFUN) + 
  theme(axis.text = element_text(color = "black")) + 
  ggtitle("Impulse reponse function (Black) \n with pointwise 95% credible range. \n Red dotted line is true lag function")

# Add the true lag function 
p + geom_line(aes(y = `lagTrue`, colour = "red"), linetype="dashed")




### Posterior fit check, comparison of fitted and unobserved (true) value 
dev.off()
post <- rstan::extract(fit3, pars = "alpha", permuted = TRUE, inc_warmup = FALSE)
mean <- apply(post$alpha, 2, mean)
lo <- apply(post$alpha, 2, quantile, 0.025)
hi <- apply(post$alpha, 2, quantile, 0.975)
plot(lo, type = "l", lty = "dotted", ylim = range(c(mean, lo, hi)), ylab = expression(alpha[t]), xlab = "Time")
points(alpha, pch = 3, cex = 0.5, col = "blue")
lines(hi, lty = "dotted")
lines(mean, type = "l", lwd = 1)
title(main = "Posterior mean of intercept (solid line) \n 95% credible interval (dotted line) \n  True intercept (cross)")





### Posterior fit check, comparison of fitted and unobserved (true) value 
post <- rstan::extract(fit3, pars = "yHat")
mean <- apply(post$yHat, 2, mean)
lo <- apply(post$yHat, 2, quantile, 0.025)
hi <- apply(post$yHat, 2, quantile, 0.975)
plot(lo, type = "l", lty = "dotted", ylim = range(c(mean, lo-1, hi +1)), ylab = expression(Y[t]), xlab = "Time")
lines(hi, lty = "dotted")
lines(mean, type = "l", lwd = 0.5)
points(y, pch = 3, cex = 0.5, col = "blue")
title(main = "Fitted mean of outcome (solid line) \n 95% credible interval (dotted line) \n  Observed outcome (cross)")



