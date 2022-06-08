
#### Script to generate sample data with time-lagged exposure-outcome association captured by transfer function implemented under dynamic reggression framework

Data are generated by R, and the model is ran by Stan software, via
rstan library. Stan performs Bayesian inference using Hamiltonian Monte
Carlo algorithms See <https://mc-stan.org/users/interfaces/rstan>

``` r
require(rstan)
require(ggplot2)

options(mc.cores=parallel::detectCores())

set.seed(101)

rm(list=ls())

# Function to generate outcome from lagged exposure-outcome association
# Function to generate matrix of exposure with lag, borrowed from : https://github.com/alastairrushworth/badlm/blob/master/R/lag_matrix.R
lag_matrix  <-  function (exposure, p, start.at.zero = T){
  windows <- function(n) rev(exposure[n:(n + p - !start.at.zero)])
  nums <- as.list(1:(length(exposure) - p))
  matrix(unlist(lapply(nums, windows)), nrow = (length(exposure) - p), ncol = p + start.at.zero, byrow = TRUE)
}
```

#### Generate a single realization of time-series, stochastic level and koyck lag

``` r
Tn <- 300 # number of time periods 


# Lag decay coefficient 
lambda <- 0.5
# Immediate effect 
beta = 1
# Observation noise, standard deviation of error for Y
sigma_Y <- 1


# Intercept, e.g.  log store-level sales of junk food, alpha_1 (initial value) is log(10)
# Standard deviation of shift in intercept 
sigma_alphaLevel <- 0.2
alpha <- cumsum(rnorm(n = Tn, sd = sigma_alphaLevel)) + 10




# Generate season wave 
weekIndex <- 1:Tn # this is time indicator, same as "t" variable in Supplemetary description of the model 
freq <- 52.2
amplitude <- 2 # To be captured by the season coefficients
w <- 2*weekIndex*pi/freq 
tempEffect <- amplitude*sin(w) + amplitude*cos(w)
# Alternatively, one can add temperature effect attached, weather in Montreal Canada between 2008 and 2013
#weather <- readRDS("./weather.rds")
#tempEffect <- as.numeric(weather$day_temp)*0.1
#tempEffect <- tempEffect[-(1:(length(tempEffect) - Tn))] # aligh number of time points 


# Exposure with lag effect 
H <- 0:20 # Horizon, max time window to define lag  
lagFunction <- beta*lambda^(H) #exponential decay 
#lagFunction <- (beta*exp(-(1-lambda)*H)) # more gradual decay
# Exposure, autoregressive order 1 
x <- arima.sim(model = list(ar = 0.6), n = Tn+max(H), sd = 0.1)
x <- scale(x)[,1]

# Lagged effect at each time period
lag_mat       <- lag_matrix(x, p = max(H))


# Generate outcome, combination of lagged effect of x, intercept, season and noise 
yMean    <- as.numeric(lag_mat %*% lagFunction) + alpha + tempEffect 
y <- yMean + rnorm(Tn, mean = 0, sd = sigma_Y)

T <- Tn # Time variable, 
x <- x[-H] # Trim lag horizon from the exposure 

### plot 
plot(lagFunction, type = "l", main = "Lag function", xlab = "Time lag", ylab = "Association")
```

![](README_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
plot(x, main = "Exposure", type = "l", xlab = "time")
```

![](README_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

``` r
plot(tempEffect, main = "Season wave (52 weeks frequency)",  type = "l", xlab = "time")
```

![](README_files/figure-gfm/unnamed-chunk-2-3.png)<!-- -->

``` r
plot(y, main = "Outcome", type = "l", xlab = "time")
```

![](README_files/figure-gfm/unnamed-chunk-2-4.png)<!-- -->

#### Fitting stan models

``` r
###Run-------------------------------------------------------------------------
# Fitting time-series model
fit3 <- stan(file="testModel_alphaLevel_season_koyck.stan", data=c("T","y","x", "weekIndex"), 
             iter=30000,  chains=3, control = list(max_treedepth = 15))

# Quick mcmc diagnosis - Sign of convergence needs to be checked before interpretation of results  
# See here: https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html

traceplot(fit3, pars = c("sigma_Y", "sigma_alpha", "beta", "lambda", "alpha[2]", "alpha[200]", "alpha[300]")) + 
  ggtitle("Trace of MCMC")
```

``` r
# Bivariate distribution of posteriors 
pairs(fit3, pars = c("sigma_Y", "sigma_alpha",  "beta", "lambda", "alpha[2]", "alpha[200]", "alpha[300]"))
```

``` r
sum(summary(fit3)$summary[,"Rhat"] > 1.01)

# further checks 
source("https://raw.githubusercontent.com/betanalpha/knitr_case_studies/master/stan_intro/stan_utility.R")
check_all_diagnostics(fit3)

# Autocorrelation function to see residual autocorrelation is not severe 
fittedPosterior <- rstan::extract(fit3, pars = "yHat")
resids <- apply(fittedPosterior$yHat, 2, mean) - y
acf(resids, main = "Autocorrelation function of residuals")
```

``` r
hist(resids, main = "Histogram of residuals")
```

#### Compile results

Analyses are valid if autocorrelation of residuals is negligible, and
for Bayesian inference, if there is a sign of convergence in MCMC

``` r
### Results --------------------------------------------------------------------

# Check estimated model paramters 
plot(fit3, 
     ci_level = 0.95,
     pars = c("beta", "lambda", "sigma_alpha", "sigma_Y", "gammaCos", "gammaSin")) + 
  ggtitle("Summary of recovered parameters (Posterior mean and 95% Interval)", 
          subtitle = 
            expression("These parameters correspond to "  ~ beta  ~ lambda  ~ sigma[alpha]  ~ sigma[epsilon]  ~ gamma[cos] ~ gamma[sin]   ~ "in Supplementary appendix"))
```

    ## ci_level: 0.95 (95% intervals)

    ## outer_level: 0.95 (95% intervals)

![](README_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
#Posterior distribution of the parameters in transfer function (1,0)  
post <- rstan::extract(fit3, pars =  c("beta", "lambda"), permuted = TRUE, inc_warmup = FALSE)
betaPosterior = post[[1]]
lambdaPosterior = post[[2]]


# Plot Impulse response function (IRF)
h <- 8 # number of lags to display, increase for long lag 
# IRF, mean, lower and uppfer CI 
irfWeight <- data.frame(weekLag = 1:h, lo = rep(NA, h), mean = rep(NA, h),  hi = rep(NA, h))
# Time t
irfWeight[1, ] <- c(1, quantile(betaPosterior , c(0.025)), mean(betaPosterior), quantile(betaPosterior , c(0.975)))
# time t+(2:h)
for(i in 2:h){irfWeight[i, ] <- c(i, quantile(beta * {lambdaPosterior^(i-1)}, c(0.025, 0.5,0.975)))} 

# Plot IRF
irfWeight$lagTrue <-  lagFunction[1:h] 
scaleFUN <- function(x) sprintf("%.1f", x)
p <-  ggplot(data = irfWeight, aes(x=(weekLag), y=mean, linetype = "est")) + 
  geom_line() + 
  theme_classic() +
  xlab("Lag") + ylab("Change of outcome") + 
  geom_ribbon(aes(ymin=lo, ymax=hi), linetype=2, alpha=0.1)  + 
  scale_y_continuous(labels=scaleFUN) + 
  theme(axis.text = element_text(color = "black")) + 
  ggtitle("Impulse reponse function (Black) \n with pointwise 95% credible range (grey band). \n compared to true lag function (red dotted line)")

# Add the true lag function 
p + geom_line(data = irfWeight, aes(x = weekLag, y = `lagTrue`, linetype="true"), ) + 
  scale_linetype_manual(name = 'Legend', values=c("est" = "solid", "true" = "dashed"), labels = c('Estimated','True'))
```

![](README_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
### Posterior fit check, comparison of fitted and true value 
post <- rstan::extract(fit3, pars = "yHat")
mean <- apply(post$yHat, 2, mean)
lo <- apply(post$yHat, 2, quantile, 0.025)
hi <- apply(post$yHat, 2, quantile, 0.975)
plot(lo, type = "l", lty = "dotted", ylim = range(c(mean, lo-1, hi +1)), ylab = expression(Y[t]), xlab = "Time")
lines(hi, lty = "dotted")
lines(mean, type = "l", lwd = 0.5)
points(y, pch = 3, cex = 0.5, col = "blue")
title(main = "Fitted mean of outcome (solid line) \n 95% credible interval (dotted line) \n  Simulated outcome (cross)")
```

![](README_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->

``` r
### Posterior fit check, comparison of fitted and unobserved (true) value of intercept 
post <- rstan::extract(fit3, pars = "alpha", permuted = TRUE, inc_warmup = FALSE)
mean <- apply(post$alpha, 2, mean)
lo <- apply(post$alpha, 2, quantile, 0.025)
hi <- apply(post$alpha, 2, quantile, 0.975)
plot(lo, type = "l", lty = "dotted", ylim = range(c(mean, lo, hi)), ylab = expression(alpha[t]), xlab = "Time")
lines(hi, lty = "dotted")
lines(mean, type = "l", lwd = 1)
points(alpha, pch = 3, cex = 0.8, col = "blue")
title(main = "Posterior mean of intercept (solid line) \n 95% credible interval (dotted line) \n  True intercept (cross)")
```

![](README_files/figure-gfm/unnamed-chunk-4-4.png)<!-- -->
