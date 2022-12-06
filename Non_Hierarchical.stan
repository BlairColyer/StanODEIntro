//the block where your ODE models are defined. Below we use one ODE equation to infer a growth rate,
//alpha and a carrying capacity, K. 
functions{ 
    vector logistic(real t, vector y, real alpha, real K){
        vector[1] dydt;
        dydt[1] = alpha * y[1] * (1 - (y[1]/K));
        return dydt;
    }
} 
//The block where we outline the inputs our model needs to infer parameters.
//note that we can truncate these using <lower=n> or <upper=n> when needed.
data{
    int<lower=1> n_rep; //Number of replicates
    int<lower=1> T; //Number of points per replicate
    array[T, n_rep] real<lower=0> y; //input data for inference
    array[n_rep] real<lower=0> y0;  //initial data point per replicate
    array[T] real<lower=0> ts; //timestep for the ODE solver
    real<lower=0> t0;  //initial time for ODE Solver
}
//this block is optional, and can be used for any reformatting of 
//data sets such as converting real values to vectors, which we do here, 
//since the ODE solver requires the initial conditions to be stored in a vector
transformed data{
    array[n_rep] vector[1] y_init;
    for(n in 1:n_rep){
        y_init[n,1]=y0[n];
    }
}
//we define our parameters here -- we have 2 in the model above which
//we store in an array, theta, and we need an additional variance parameter, 
//sigma, for inference later. 
parameters{
    array[2] real <lower=0> theta; //alpha and K 
    real<lower=0> sigma;
}
//this is where we approximate our ODE solutions to compare to our data later. 
//the actual sampling happens in the next block.
transformed parameters{
    array[T] vector[n_rep] yhat;
    for (n in 1:n_rep){
        array[T] vector[1] yhat_temp;
        yhat_temp = ode_rk45(logistic, y_init[n,:], t0, ts, theta[1], theta[2]);
        for(t in 1:T){
            yhat[t,n]=yhat_temp[t,1]; //here we just store the ODE solutions for later
        }
    }
}
//our final block here is where we do our sampling. We have informative prior distributions
//for our parameters here, but since the data was generated using an exact model solution with noise, 
//we already know what these distributions should be (more or less). Typically this is where you
//would add any information gleaned from preliminary data analysis. The more informative your prior, the
//better your inference will be. 
model{ 
    theta[1] ~ normal(1,1); //alpha
    theta[2] ~ normal(100,1); //K. These are used in the ODE approximation
    sigma ~ normal(0,1); //extra variance parameter which is used below. 
    for(n in 1:n_rep){
        for(t in 1:T){
            y[t,n] ~ normal(yhat[t,n], sigma); //this is our sampling step where 
            // data is compared to approximated solutions.

        }
    }
}

//we must always leave a blank line at the end -- this is just to do
//with how Stan compiles models. This is all we need for inference and 
//we could do it in about 40 lines of code (excluding annotations). 

// A final note -- if you see any issues with this please let me know!