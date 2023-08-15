#  Censored Neural Estimators

Scripts to perform point estimation with neural Bayes estimators and censored input data. 


The code in `src` reproduces training and assessment of two of the estimators used by Richards et al. (2023+):

* `HW_cNBE.jl`: this script constructs estimators for the random scale mixture process used by Richards et al. (2023+) to model extreme Arabian PM2.5 concentrations. Training is conducted using "simulation-on-the-fly", and is done solely via this Julia script. Due to storage constraints, we only provide here the standardised data with `G=4` used in the application. Whilst the code will run for the other values of `G` considered by Richards et al. (2023+), the required data must be downloaded from <u><a href="https://urldefense.com/v3/__https://doi.org/10.5281/zenodo.8246931__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjaqNwWhfg$ 
">Zenodo</a></u> and placed in `data/`. The original PM2.5 concentrations data are provided in `data/PM25_original.Rdata`.
* `simulate_IMSP_random_tau.R`, followed by `IMSP_random_tau_cNBE.jl`: reproduces the "random-tau" estimator for the inverted max-stable process (IMSP; see Section 4.3 of Richards et al., 2023+). Simulation of the IMSP is first carried out in R using `simulate_IMSP_random_tau.R`. We then pass these replicates to Julia and train the estimator using `IMSP_random_tau_cNBE.jl`.

Note that the scripts are designed to run quickly; to fully replicate the estimators used by Richards et al. (2023+), certain arguments must be changed (detailed comments are included in the scripts).
For details on general neural Bayes estimators, see Sainsbury-Dale et al. (2023+). The scripts in this repository depend on the [NeuralEstimators](https://urldefense.com/v3/__https://github.com/msainsburydale/NeuralEstimators.jl__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjYMNUBtJQ$ ) Julia package.

## Installation 
To run the scripts, you must first install the Julia and R package dependencies. This can be achieved by navigating to the top level of the repository in the terminal and then running:

 - `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`. This will install all Julia package dependencies given in `Project.toml` and `Manifest.toml`.
 - `Rscript Dependencies.R`. This will install all R package dependencies.

## References 
<ul> 
          <li> Richards, J., Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Neural Bayes estimators for censored inference with peaks-over-threshold models. <u><a href="https://urldefense.com/v3/__https://arxiv.org/abs/2306.15642__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjYzywvTTw$ " download>arxiv.org/2306.15642</a></u> </li>
          <li> Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Likelihood-Free Parameter Estimation with Neural Bayes Estimators. <u><a href="https://urldefense.com/v3/__https://arxiv.org/abs/2208.12942__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjbQN-_SqQ$ " download>arxiv.org/2208.12942</a></u> </li>
</ul>
