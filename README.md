#  Censored Neural Estimators

Scripts to perform point estimation with neural Bayes estimators and censored input data. The code reproduces the estimators for the random scale mixture process (with `G=4`) used by Richards et al. (2023+) to model extreme Arabian PM2.5 concentrations. Note that due to storage limitations, we can only provide here the data with `G=4`. Whilst the code runs for the other values of `G` considered by Richards et al. (2023+), these data must be acquired from the authors.

For details on neural Bayes estimators, see Sainsbury-Dale et al. (2023+). The scripts in this repository depend on the [NeuralEstimators](https://github.com/msainsburydale/NeuralEstimators.jl) Julia package. An older version of this package is stored within this repository.


## Installation 
To run the script, you must first install `NeuralEstimators` from source.

1. Install the `Julia` version of `NeuralEstimators`.
	- To install from terminal, run the command `julia -e 'using Pkg; Pkg.add(PackageSpec(path="~/CensoredNeuralEstimators.jl/NeuralEstimators.jl"))'`.
1. Install the deep-learning library `Flux` and other pre-requisite packages.
	- To install from terminal, run the command `julia -e 'using Pkg; Pkg.add(["Flux","Distances","Folds","LinearAlgebra","Distributions","CSV","Tables","Random",'Rdata"])'`

## References 
<ul> 
          <li> Richards, J., Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Likelihood-free neural Bayes estimators for censored inference with peaks-over-threshold models. <u><a href="https://arxiv.org/abs/2306.15642" download>arxiv.org/2306.15642</a></u> </li>
          <li> Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Likelihood-Free Parameter Estimation with Neural Bayes Estimators. <u><a href="https://arxiv.org/abs/2208.12942" download>arxiv.org/2208.12942</a></u> </li>
</ul>
