### Dynamic linar model with transfer function (1,0) to capture monotonic lag decay, including time-varying intercept, stochastic level 
# Use rstan for inference 
# Hiroshi Mamiya 
# July/2021

require(rstan)
require(ggplot2)

options(mc.cores=parallel::detectCores())

rm(list=ls())
set.seed(1010)

setwd()

# Numer of observations (time points)
Tn <- 300

#  Lag coefficient
lambda <- 0.5 

# Immediate effect 
beta = 0.3

# Observation noise, standard deviation of error for Y 
sigma_Y <- 0.5

# Standard deviation of shift in intercept 
sigma_alphaLevel <- 0.1



# Intercept, e.g.  log store-level sales of junk food, alpha_1 (initial value) is log(10)
alpha <- cumsum(rnorm(n = Tn, sd = sigma_alphaLevel)) + 10
plot(alpha, main = "Intercept, random walk without trend")


# Lag function of exposure  
H <- 0:20 # lag horizen 
lagFunction <- (beta*exp(-(1-lambda)*H))
plot(lagFunction, main = "A monotonic decay \n e.g., advertising-sales or heatwave-mortality association")

# Exposure, ar(1) - centered and scaled
x <- arima.sim(model = list(ar = 0.6), n = Tn+max(H), sd = 0.1)
x <- scale(x)[,1]
plot(x, main = "Exposure")

# Create lag 
# Function to generate matrix of exposure with lag, borrowed from : https://github.com/alastairrushworth/badlm/blob/master/R/lag_matrix.R
lag_matrix	<-	function (rain, p, start.at.zero = T){
  slider <- function(n) rev(rain[n:(n + p - !start.at.zero)])
  nums <- as.list(1:(length(rain) - p))
  matrix(unlist(lapply(nums, slider)), nrow = (length(rain) - p), ncol = p + start.at.zero, byrow = TRUE)
}

lag_mat       <- lag_matrix(x, p = max(H))

# Generate outcome, combination of lagged effect of x, intercept and noise 
yMean    <- as.numeric(lag_mat %*% lagFunction) + alpha
plot(yMean); 
lines(alpha, type = "l"); 
title(main = "Mean of Y(circle) and intercept (line)")

# Add noise
y <- yMean + rnorm(Tn, mean = 0, sd = sigma_Y)





### Compile and execute a Stan model 
T <- Tn # Time variable, not boolian 
x <- x[-H] # Trim lag horizon

# Fixed intercept, no level 
fit1 <- stan(file="testModel_koyck.stan", data=c("T","y","x"), 
            iter=10000,  chains=3, control = list(max_treedepth = 15, adapt_delta = 0.95))
# Diagnosis
traceplot(fit1, pars = c("sigma_V", "beta", "lambda", "alpha"))
pairs(fit1, pars = c("sigma_V", "beta", "lambda", "alpha"))
sum(summary(fit1)$summary[,"Rhat"] > 1.01)


# Fitting dynamic linar model, state vector (alpha) written in the transformed parameter block 
fit2 <- stan(file="testModel_alphaLevel_koyck_reparam.stan", data=c("T","y","x"), 
            iter=10000,  chains=3, control = list(max_treedepth = 15, adapt_delta = 0.95))
# Diagnosis
traceplot(fit2, pars = c("sigma_V", "sigma_alpha[1]", "sigma_alpha[2]", "scale_alpha", "beta", "lambda", "alpha[1]", "alpha[200]", "alpha[300]"))
pairs(fit2, pars = c("sigma_V", "sigma_alpha[1]", "sigma_alpha[2]", "scale_alpha", "beta", "lambda", "alpha[1]", "alpha[200]", "alpha[300]"))
sum(summary(fit2)$summary[,"Rhat"] > 1.01)


# Fitting dynamic linar model, state vector (alpha) written in the model block   
fit3 <- stan(file="testModel_alphaLevel_koyck.stan", data=c("T","y","x"), 
            iter=10000,  chains=3, control = list(max_treedepth = 15))
#  Diagnosis
traceplot(fit3, pars = c("sigma_V", "sigma_alpha", "beta", "lambda", "alpha[1]", "alpha[200]", "alpha[300]"))
pairs(fit3, pars = c("sigma_V", "sigma_alpha",  "sigma_alpha", "beta", "lambda", "alpha[1]", "alpha[200]", "alpha[300]"))
sum(summary(fit3)$summary[,"Rhat"] > 1.01)




# Interpretation below is relevant only if signs of convergence in MCMC are observed
# Only focusing on the parameters from the fit 3 above


# Histogram ofposterior distribtion compared with true value 
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
plotList[[1]] <- plotHist(fit3,  param = "lambda", trueValue = lambda, range = c(0,1), label = expression(lambda))
plotList[[2]] <- plotHist(fit3,  param = "beta", trueValue = beta, label = expression(beta))
plotList[[3]] <- plotHist(fit3,  param = "sigma_alpha", trueValue = sigma_alphaLevel, label = expression(sigma[alpha]))
plotList[[4]] <- plotHist(fit3,  param = "sigma_V", trueValue = sigma_Y, label = expression(sigma[Y]))

cowplot::plot_grid(plotlist = plotList, ncol = 2, labels = "auto", label_fontface = "plain")




### Generate Impuse response function from fit 
post <- rstan::extract(fit3, pars =  c("beta", "lambda"), permuted = TRUE, inc_warmup = FALSE)
betaPosterior = post[[1]]; lambdaPosterior = post[[2]]
h <- 8
# IRF, mean, lower and uppfer CI 
irfWeight <- data.frame(weekLag = 1:h, lo = rep(NA, h), median = rep(NA, h),  hi = rep(NA, h))
# Time t
irfWeight[1, ] <- c(1, quantile(betaPosterior , c(0.025)), median(betaPosterior), quantile(betaPosterior , c(0.975)))
# time t+(2:h)
for(i in 2:h){irfWeight[i, ] <- c(i, quantile(beta * {lambdaPosterior^(i-1)}, c(0.025, 0.5,0.975)))} 

# Plot Impulse response function 
irfWeight$lagTrue <-  lagFunction[1:h] 
  
scaleFUN <- function(x) sprintf("%.1f", x)
p <-  ggplot(data = irfWeight, aes(x=(weekLag), y=median)) + 
  geom_line() + 
  theme_classic() +
  xlab("Lag") + ylab("Change of outcome") + 
  geom_ribbon(aes(ymin=lo, ymax=hi), linetype=2, alpha=0.1)  + 
  scale_y_continuous(labels=scaleFUN) + 
  theme(axis.text = element_text(color = "black")) + 
  ggtitle("Impulse reponse function (Black) and pointwise 95% credible range.")

# Add the true lag function 
p + geom_line(aes(y = lagTrue, colour = "red"), linetype="dashed")

 
# Posteior fit check, comparison of fitted and unobserved (true) value 
 params <- rstan::extract(fit3, pars = "yHat")
 median <- apply(params$yHat, 2, median)
 lo <- apply(params$yHat, 2, quantile, 0.025)
 hi <- apply(params$yHat, 2, quantile, 0.975)
 plot(lo, type = "l", lty = "dotted", ylim = range(c(median, lo, hi)), ylab = expression(Y[t]), xlab = "Time")
 points(y, pch = 3, cex = 0.5, col = "blue")
 lines(hi, lty = "dotted")
 lines(median, type = "l", lwd = 1)
 title(main = "Predictive distribution of and observed Y \n Fitted mean:solid line, 95% credible interval: dotted line, overved Y: cross ")
 
# Compare posteior of alpha and true values  
params <- rstan::extract(fit3, pars = "alpha")
mean <- apply(params$alpha, 2, mean)
lo <- apply(params$alpha, 2, quantile, 0.025)
hi <- apply(params$alpha, 2, quantile, 0.975)
plot(lo, type = "l", lty = "dotted", ylim = range(c(mean, lo, hi)), ylab = expression(alpha[t]), xlab = "Time")
points(alpha, pch = 3, cex = 0.5, col = "blue")
lines(hi, lty = "dotted")
lines(mean, type = "l", lwd = 1)
title(main = "Estimated (solid line) and true (dot) value of stochastic intercept \n and 95% credible interval (dotted line)")








