# Installing packages we use for data organising, inference and visualisation

library(tidyverse)
library(gridExtra)
library(bayesplot)
library(cmdstanr)
library(posterior)
library(shinystan)
options(mc.cores = parallel::detectCores())

#locate the data file directory below
data <- read.csv(".../10Dataset10Datapoints.csv",
    header = TRUE, stringsAsFactors = FALSE)

 #renames the data columns -- not strictly necessary
data <- rename(data, time = X, r0 = X0, r1 = X1, r2 = X2, r3 = X3,
    r4 = X4,  r5 = X5,  r6 = X6, r7 = X7,  r8 = X8,  r9 = X9)


#preparing the data we need to input into our model data for stan in
#a format it can work with
T <- length(data$time) - 1
y0 <- filter(data, time == 0) %>% select(-time) %>% unlist
y0 <- as.vector(y0)
t0 <- 0.0
ts <- 1:T
y <- filter(data, time > 0) %>% select(-time)
n_rep <- 10
model_data <- list(n_rep = n_rep, T = T, y = y, y0 = y0, ts = ts, t0 = t0)

##navigate to the directory the stan file is located in
mod_SNH <- cmdstan_model(".../Standard_Non_Hierarchical.stan")

##produces executable for the inference
mod_SNH$exe_file()

#takes the model data and runs the inference with 4 chains in
#parallel. Adept delta is at default value for this
#run, since the model (SHOULD) run without divergence.

fit_SNH <- mod_SNH$sample(data = model_data, seed = 123,
     chains = 4, iter_warmup = 500,
     iter_sampling = 2000, refresh = 200,
     adapt_delta = 0.8, max_treedepth = 15)

#this will show you your results along with
#the number of effective samples and the Rhat
#value, which should be <1.01 if your chains
#have mixed properly.
print(fit_SNH, max_rows = 200)

#I also include stan code for Hierarchical models,
#using both centered and non-centered parameterisations.
#We have enough data here that the centered param.
#should work just fine, but I include the non-centered
#one based on the assumption that some of us won't have
#access to this many experimental replicates.

#Below we have the centered hierarchical model, which
#fits a set of parameters to each replicate. See the
#file itself for more details.
mod_CH <- cmdstan_model(".../Centered_Hierarchical.stan")

mod_CH$exe_file()

fit_CH <- mod_CH$sample(data = model_data, seed = 123, chains=4,
    iter_warmup = 500, iter_sampling = 2000,
    refresh = 200, adapt_delta = 0.8, max_treedepth = 15)
#we have a few divergences here -- we'll run this again with
#a higher adapt_delta value.
fit_CH <- mod_CH$sample(data = model_data, seed = 123, chains = 4,
    iter_warmup = 500, iter_sampling = 2000,
    refresh = 200, adapt_delta = 0.99, max_treedepth = 15)

print(fit_CH, max_rows = 200)
#We're still seeing divergences, albeit fewer. We know our priors are good,
#increasing adapt delta didn't work, so now, we'll try to
#reparameterise the model.
mod_NCH <- cmdstan_model(".../Non_Centered_Hierarchical.stan")

mod_NCH$exe_file()

fit_NCH <- mod_NCH$sample(data = model_data, seed = 123, chains = 4,
     iter_warmup = 500, iter_sampling = 2000, refresh = 200,
     adapt_delta = 0.8, max_treedepth = 15)

#success! we have no divergences, the Rhat measure
#shows the chains are well mixed and the number of
#effective samples is good!

print(fit_NCH, max_rows = 200)
