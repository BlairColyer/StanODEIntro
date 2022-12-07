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
//this is where we see the first difference between the standard model
//and the hierarchical -- we have 5 extra parameters here.

//(I strongly suggest doing some background reading here -- 
//there are a lot of moving parts but they make sense when we
//see what they're doing in the model. 
//I describe each one briefly, but a lot of them, like sigma for the last model,
//are just parameters we need to define for the inference model. 
parameters{
    real<lower=0, upper=4> mu_theta_alpha; //hierarchical mean's mean for alpha
    real<lower=90,upper=110> mu_theta_K; //hierarchical mean's mean for K
    array[n_rep,2] real <lower=0> theta; //Group level parameter means
    array[2] real<lower=0,upper=4> tau; //Hierarchical mean's variance
    real<lower=0> mu_sigma; //Hierarchical variance's mean
    real<lower=0> xi; //hierarchical variance's variance
    array[n_rep] real<lower=0> sigma; //group level variance.
}
transformed parameters{
    array[2] real mu_theta;
        mu_theta[1]=mu_theta_alpha;
        mu_theta[2]=mu_theta_K;
    array[T] vector[n_rep] yhat;
    for (n in 1:n_rep){
        array[T] vector[1] yhat_temp;
        yhat_temp = ode_rk45(logistic, y_init[n,:], t0, ts, theta[n,1], theta[n,2]);
        for(t in 1:T){
            yhat[t,n]=yhat_temp[t,1];
        }
    }
}
//this is the only other place with major differences to the original model. 

model{
    mu_theta[1] ~ normal(1,1); //Hierarchical parameters sampled
    mu_theta[2] ~ normal(100,1); 

    tau[1] ~ normal(0,1); //Variances sampled for the Hierarchical means.
    tau[2] ~ normal(0,1); // we see them used below.

    mu_sigma ~ normal(0,0.1); //Variance mean for hierarchical variance
    xi ~ normal(0,0.1); //variance for this variance.

    //The next part is fairly straightforward, and hopefully makes it clearer
    //where some of the above values are coming from and how we use them.

    //the array theta contains 1 alpha and K value per replicate. 
    //rather than sampling these directly, we use the values sampled above
    //so that each replicate shares the hierarchical distribution. 
    // to sample from this we need both a mean (mu theta) and variance (tau).

    //the array sigma is then used below in the section where data is compared to 
    //approximated solutions in the inference step, but each replicate gets its
    //own variance value, too. to sample this, again, we need a mean (mu_sigma) and 
    //a variance (xi) which each replicate shares. 
    for (n in 1:n_rep){
        for (j in 1:2){
            theta[n,j] ~ normal(mu_theta[j],tau[j]);
        }
        sigma[n] ~ normal(mu_sigma,xi); 
    }
    for(n in 1:n_rep){
        for(t in 1:T){
            y[t,n] ~ normal(yhat[t,n], sigma[n]);
        }
    }
}
