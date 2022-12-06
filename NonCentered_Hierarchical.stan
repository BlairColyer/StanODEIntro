functions{ 
    vector logistic(real t, vector y, real alpha, real K){
        vector[1] dydt;
        dydt[1] = alpha * y[1] * (1 - (y[1]/K));
        return dydt;
    }
} 
data{
    int<lower=1> n_rep; 
    int<lower=1> T; 
    array[T, n_rep] real<lower=0> y; 
    array[n_rep] real<lower=0> y0;  
    array[T] real<lower=0> ts;
    real<lower=0> t0;
}
transformed data{
    array[n_rep] vector[1] y_init;
    for(n in 1:n_rep){
        y_init[n,1]=y0[n];
    }
}
//We have the same hierarchical parameters, but now we have
//two new parameters, theta_tilde and sigma_tilde. These are latent
//gaussian variables which we use to recover the group level 
//parameters with a scaling and a translation in the transformed parameter
//block. 
parameters{
    real<lower=0, upper=4> mu_theta_alpha; 
    real<lower=90,upper=110> mu_theta_K;
    array[2] real<lower=0,upper=4> tau; 
    array[n_rep, 2] real theta_tilde;
    real<lower=0> mu_sigma; 
    real<lower=0> xi;
    array[n_rep] real<lower=0> sigma_tilde;
}
//the first for loop in this block is where we implement the non-centered
//parameterisation of the model. Below we will see that theta_tilde 
//and sigma_tilde are just values sampled from normal(0,1). We multiply
//this value by the relevant variance from tau, and then add the relevant
//hierarchical mean from mu_theta to simulate a random sample from a normal
//distribution. Simulating different distributions in this way will require
//different translations of these parameters! 
transformed parameters{
    array[2] real mu_theta;
        mu_theta[1]=mu_theta_alpha;
        mu_theta[2]=mu_theta_K;
    array[n_rep, 2] real theta;
    array[n_rep] real sigma;
    array[T] vector[n_rep] yhat;
    for (n in 1:n_rep){
        for (j in 1:2){
            theta[n, j] = mu_theta[j] + tau[j] * theta_tilde[n,j];
        }
        sigma[n] = mu_sigma + xi * sigma_tilde[n];
    }
    for (n in 1:n_rep){
        array[T] vector[1] yhat_temp;
        yhat_temp = ode_rk45(logistic, y_init[n,:], t0, ts, theta[n,1], theta[n,2]);
        for(t in 1:T){
            yhat[t,n]=yhat_temp[t,1];
        }
    }
}
//things here are largely the same -- we just sample theta tilde
//and sigma tilde rather than theta and sigma, and transform them above. 
model{
    mu_theta_alpha ~ normal(1,1); 
    mu_theta_K ~ normal(100,1);
    tau[1] ~ normal(0,1);
    tau[2] ~ normal(0,1);
    mu_sigma ~ normal(0,0.1);
    xi ~ normal(0,0.1);
    for (n in 1:n_rep){
        for (j in 1:2){
            theta_tilde[n,j] ~ normal(0,1);
            sigma_tilde[n] ~ normal(0,1); 
        }
    }
    for(n in 1:n_rep){
        for(t in 1:T){
            y[t,n] ~ normal(yhat[t,n], sigma[n]);
        }
    }
}
