
require("fields")
require("SpatialExtremes")
require("abind")
require("parallel")


coords=expand.grid(1:16,1:16) #Create spatial grid and distance matrix
h=rdist(coords)

# Prior for parameters
prior <- function(K) {
  theta <- matrix(0, nrow=2, ncol = K)
  for(i in 1:K){
    range    <- runif(1,2,10)
    smooth <- runif(1, 0.5,2)
    theta[,i] <- c(range, smooth)
  }
  return(theta)
}

K=1000 #In paper, K = 125000

#Draw training/validation/test parameter sets
set.seed(1)
theta_train = prior(K)
set.seed(2)
theta_val   = prior(K/5)
set.seed(3)
theta_test  = prior(1000)

#Prior for tau
tau_prior <- function(K) {
  theta <- matrix(0, nrow=1, ncol = K)
  for(i in 1:K){
    p    <- runif(1,0.85,0.95)
    theta[,i] <- c(p)
  }
  return(theta)
}


#Drawn K random taus
set.seed(1)
tau_train = tau_prior(K)
set.seed(2)
tau_val   = tau_prior(K/5)
set.seed(3)
tau_test  = tau_prior(1000)

#Simulation done in parallel
ncores <- 24
cl <- makeCluster(ncores)
setDefaultCluster(cl=cl)


invisible(clusterEvalQ (cl , library("abind")))
invisible(clusterEvalQ (cl , library("fields")))
invisible(clusterEvalQ (cl , library("SpatialExtremes")))
clusterExport(cl , "coords")

# ---- Process simulation ----

c_tau <- 0 ; clusterExport(cl , "c_tau")  # Censored values will be set to this constant.
# Note that the standard marginal distribution function is taken to be F_* ∼ Exp(1).
# This can be changed within the following function 'simulate', but is not automatically implemented through a functional argument.

# Simulation function
simulate <- function(theta_set, tau_set,m) { #m in number of replicates


	theta_set=lapply(seq_len(ncol(theta_set)), function(i) c(theta_set[,i],tau_set[,i] ))


		parLapply(theta_set, cl=cl, fun=function(theta) {

		N <- nrow(coords)
		tau <- theta[3]

		Z <-t(rmaxstab(m,coord=coords,grid=F,cov.mod = "brown",range=theta[1],smooth=theta[2]))
		Z <- 1/Z #Z is now on standard exponential margins - a transformation can be applied here to change F_*
		# i.e., Z <- qnorm(pexp(Z)) for Gaussian margins
		censor.threshold = qexp(tau) #Censoring threshold taken to be tau-quantile for standard exponential
		Z[Z <= censor.threshold]<- c_tau
		dim(Z)=c(sqrt(N),sqrt(N),1,m)

		Z<-abind(Z,1*(Z > censor.threshold),along=3)

		Z
	})


}


#Simulate 200 replicates
m <- 200
clusterExport(cl , "m")

# Draw Z_train/Z_val/Z_test
invisible(clusterEvalQ (cl , set.seed(1)))
Z_train <- simulate(theta_train,tau_train, m)
invisible(clusterEvalQ (cl , set.seed(2)))
Z_val   <- simulate(theta_val,tau_val, m)
invisible(clusterEvalQ (cl , set.seed(3)))
Z_test   <- simulate(theta_test,tau_test, m)


#Save replicates to pass to NBE in Julia
save(Z_train,Z_val,Z_test,theta_train,theta_test,theta_val,m,coords,tau_train,tau_val,tau_test,
     file=paste0("training_replicates/IMSP_reps.Rdata"))


#We also save ten sets of single parameter test data
invisible(clusterEvalQ (cl , set.seed(1)))



for(i in 1:10){
  set.seed(i+100)
  theta_test <- matrix(rep(prior(1),1000),nrow=2)
  tau_test <- matrix(rep(tau_prior(1),1000),nrow=1)


   Z_test   <- simulate(theta_test,tau_test, m)

  save(theta_test,tau_test,Z_test, coords, file=paste0("training_replicates/IMSP_test_reps",i,".Rdata"))

}



stopCluster(cl)


