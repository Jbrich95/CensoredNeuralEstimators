#  Censored Neural Estimators

Code to perform point estimation with neural Bayes estimators and censored input data based on the methodology described in "Neural Bayes estimators for censored inference with peaks-over-threshold models" ([Richards et al., 2023+](https://urldefense.com/v3/__https://arxiv.org/abs/2306.15642__;!!Nmw4Hv0!xmRvCNlB05kCY-VXdaeczmzYcfzVGYUXX_VQPpW-OnVNw0rs-Hgy_8QtK214VbIkK9BX_aXwGq63LQ2Jm_5hbsH1zw$ )). See also [Sainsbury-Dale et al. (2023+)](https://urldefense.com/v3/__https://arxiv.org/abs/2208.12942__;!!Nmw4Hv0!xmRvCNlB05kCY-VXdaeczmzYcfzVGYUXX_VQPpW-OnVNw0rs-Hgy_8QtK214VbIkK9BX_aXwGq63LQ2Jm_58jOxElw$ ) for a general overview of neural Bayes estimation.

The scripts in this repository are based primarily on the Julia package [NeuralEstimators](https://urldefense.com/v3/__https://github.com/msainsburydale/NeuralEstimators.jl__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjYMNUBtJQ$ ) and its accompanying [R interface](https://urldefense.com/v3/__https://github.com/msainsburydale/NeuralEstimators.git__;!!Nmw4Hv0!xmRvCNlB05kCY-VXdaeczmzYcfzVGYUXX_VQPpW-OnVNw0rs-Hgy_8QtK214VbIkK9BX_aXwGq63LQ2Jm_7-3oeasQ$ ).

<!-- The methodology described in the manuscript has been incorporated into the user-friendly and well-documented Julia package, [NeuralEstimators.jl](https://urldefense.com/v3/__https://github.com/msainsburydale/NeuralEstimators.jl__;!!Nmw4Hv0!xmRvCNlB05kCY-VXdaeczmzYcfzVGYUXX_VQPpW-OnVNw0rs-Hgy_8QtK214VbIkK9BX_aXwGq63LQ2Jm_5uh2hO9A$ ), and its accompanying [R interface](https://urldefense.com/v3/__https://github.com/msainsburydale/NeuralEstimators__;!!Nmw4Hv0!xmRvCNlB05kCY-VXdaeczmzYcfzVGYUXX_VQPpW-OnVNw0rs-Hgy_8QtK214VbIkK9BX_aXwGq63LQ2Jm_4mwELynA$ ). The code in this repository is made available primarily for reproducibility purposes, and we encourage readers seeking to implement GNN-based neural Bayes estimators to explore the package and its documentation.  -->


## Installation

First, download this repository and navigate to its top-level directory within terminal.

Before installing the software dependencies, users may wish to setup a [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment, so that the dependencies of this repository do not affect the user's current installation. To create a `conda` environment, run the following command in terminal:

```
conda create -n CensoredNeuralEstimators -c conda-forge julia=1.8.0 r-base nlopt
```

Then activate the `conda` environment with:

```
conda activate CensoredNeuralEstimators
```

The above `conda` environment installs Julia and R automatically. If you do not wish to use a `conda` environment, you will need to install Julia and R manually if they are not already on your system:  

- Install Julia 1.8.0. (See [here](https://urldefense.com/v3/__https://julialang.org/downloads/__;!!Nmw4Hv0!xmRvCNlB05kCY-VXdaeczmzYcfzVGYUXX_VQPpW-OnVNw0rs-Hgy_8QtK214VbIkK9BX_aXwGq63LQ2Jm_5MhM8zTg$ ).)
- Install R >= 4.0.0. (See [here](https://urldefense.com/v3/__https://www.r-project.org/__;!!Nmw4Hv0!xmRvCNlB05kCY-VXdaeczmzYcfzVGYUXX_VQPpW-OnVNw0rs-Hgy_8QtK214VbIkK9BX_aXwGq63LQ2Jm_5875q9TQ$ ).)

Once Julia and R are setup, install package dependencies by running:

- `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`. This will install all Julia package dependencies given in `Project.toml` and `Manifest.toml`.
- `Rscript Dependencies.R`. This will install all R package dependencies.

## Reproducing the results

The code in `src` reproduces training and assessment of two of the estimators used by Richards et al. (2023+):

* `HW_cNBE.jl`: this script constructs estimators for the random scale mixture process used by Richards et al. (2023+) to model extreme Arabian PM2.5 concentrations. Training is conducted using "simulation-on-the-fly", and is done solely via this Julia script. Due to GitHub's requirement that individual files do not exceed 50MB in size, we only provide here the standardised data with `G=4` used in the application. Whilst the code will run for the other values of `G` considered by Richards et al. (2023+), the required data must be downloaded from <u><a href="https://urldefense.com/v3/__https://doi.org/10.5281/zenodo.8246931__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjaqNwWhfg$
">Zenodo</a></u> and placed in `data/`. The original PM2.5 concentrations data are provided in `data/PM25_original.Rdata`.
* `simulate_IMSP_random_tau.R`, followed by `IMSP_random_tau_cNBE.jl`: reproduces the "random-tau" estimator for the inverted max-stable process (IMSP; see Section 4.3 of Richards et al., 2023+). Simulation of the IMSP is first carried out in R using `simulate_IMSP_random_tau.R`. We then pass these replicates to Julia and train the estimator using `IMSP_random_tau_cNBE.jl`.

Note that the scripts are designed to run quickly using a very small number of sampled parameter vectors during training (i.e., setting 'K' in the manuscript to be very small); to fully replicate the estimators used by Richards et al. (2023+), this value must be changed (detailed comments are included in the scripts).

Julia must be started with the command `julia --project=.` in order to load an environment with the correct package dependencies. Users wishing to run the scripts from the terminal may enter the following commands:
* `julia --project=. --threads=auto src/HW_cNBE.jl`
* `Rscript src/simulate_IMSP_random_tau.R`
* `julia --project=. --threads=auto src/IMSP_random_tau_cNBE.jl`


## References
<ul>
          <li> Richards, J., Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Neural Bayes estimators for censored inference with peaks-over-threshold models. <u><a href="https://urldefense.com/v3/__https://arxiv.org/abs/2306.15642__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjYzywvTTw$ " download>arxiv.org/2306.15642</a></u> </li>
          <li> Sainsbury-Dale, M., Zammit-Mangion, A., and Huser, R. (2023+). Likelihood-free parameter estimation with neural Bayes estimators. <i> The American Statistician </i> (to appear). <u><a href="https://urldefense.com/v3/__https://arxiv.org/abs/2208.12942__;!!Nmw4Hv0!2MGmd5XilWCrD15Y3NNPSxQnfZwq3X3suz-Fo0QcgMAwD_RfkmHog2Y6oLcsorucWfVVJSi1kMhlXncLVjbQN-_SqQ$ " download>arxiv.org/2208.12942</a></u> </li>
</ul>
