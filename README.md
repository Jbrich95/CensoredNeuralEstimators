#  Censored Neural Estimators

Scripts to perform point estimation with neural Bayes estimators and censored input data. 


The code in `scripts` reproduces training and asessment of two of the estimators used by Richards et al. (2023+):

* `HW_cNBE.jl`: this script constructs estimators for the random scale mixture process (with `G=4`) used by Richards et al. (2023+) to model extreme Arabian PM2.5 concentrations. Training is conducted using "simulation-on-the-fly", and is done solely via this Julia script. Due to storage constraints, we only provide here the standardised data with `G=4`. Whilst the code will run for the other values of `G` considered by Richards et al. (2023+), the required data must be acquired from the authors. The original PM2.5 concentrations data are provided in `data/PM2.5 concentrations.Rdata`.
* `simulate_IMSP_random_tau.R`, followed by `IMSP_random_tau_cNBE.jl`: reproduces the "random tau" estimator for the inverted max-stable process (IMSP; see Section 4.3 of Richards et al., 2023+). Simulation of the IMSP is first carried out in R using `simulate_IMSP_random_tau.R`. We then pass these replicates to Julia and train the estimator using `IMSP_random_tau_cNBE.jl`.

For details on general neural Bayes estimators, see Sainsbury-Dale et al. (2023+). The scripts in this repository depend on the [NeuralEstimators](https://github.com/msainsburydale/NeuralEstimators.jl) Julia package. An older version of this package is stored within this repository.


## Installation 
To run the script, you must first install `NeuralEstimators` from source.

1. Install the `Julia` version of `NeuralEstimators`.
	- To install from terminal, run the command `julia -e 'using Pkg; Pkg.add(path="NeuralEstimators.jl")'` when in repository directory.
1. Install the deep-learning library `Flux` and other pre-requisite packages.
	- To install from terminal, run the command `julia -e 'using Pkg; Pkg.add(["Flux","Distances","Folds","LinearAlgebra","Distributions","CSV","Tables","Random","RData"])'`

## References 
<ul> 
          <li> Richards, J., Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Likelihood-free neural Bayes estimators for censored inference with peaks-over-threshold models. <u><a href="https://arxiv.org/abs/2306.15642" download>arxiv.org/2306.15642</a></u> </li>
          <li> Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Likelihood-Free Parameter Estimation with Neural Bayes Estimators. <u><a href="https://arxiv.org/abs/2208.12942" download>arxiv.org/2208.12942</a></u> </li>
</ul>
