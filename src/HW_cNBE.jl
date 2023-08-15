using NeuralEstimators
using Distances: pairwise, Euclidean, Haversine
using Folds
using LinearAlgebra
using Distributions

using Flux
using CSV
using Tables
using Random: seed!
using RData


τ = 0.9 # Set censoring level

G = 4 # Set smoothing level. For G > 4, data must be downloaded from https://zenodo.org/record/8246931.


# Set number of training samples K. Here we use K = 500 to ensure that the code will run on a personal computer. 
# To reproduce results fully, see details in paper for values of K used for different G. Note that using the same values of K from the paper will lead to memory problems on most personal machines.

K = 500 



# Set the prior distribution
Ω = (
	λ = Uniform(20.0, 1250.0),
	δ = Uniform(0.0, 1.0),
	ν = Uniform(0.1, 4.0),
	L = Uniform(0.5,3.5),
	ω = Uniform(-pi/2,0.0)
)

# Set your reference spatial domain and calculate pairwise distances. We take the reference domain to be a G by G grid that is roughly in the centre of the application domain.

S = expandgrid(LinRange(45.05,45.05+(G-1)*0.1, G), LinRange(23.75,23.75+(G-1)*0.1, G))
D = pairwise(Haversine(6378.388), S,S , dims = 1)


# Tuple to store the important model information
ξ = (
	Ω = Ω,
	S = S,
	p = length(Ω),
	d = size(S, 1),
	parameter_names = String.(collect(keys(Ω))),
	side_length = G
)

# Define the anisotropic transformation
function anisotransform(s, ω, L)
  Lmat = Diagonal([1.0, 1/L])
  ψmat = [cos(ω) -sin(ω); sin(ω) cos(ω)] # this notation fills a matrix by row: see [1 2; 3 4]
  Lmat * ψmat * s
end

# ---- Sampling parameters ----

struct Parameters{T, V, J} <: ParameterConfigurations
	θ::Matrix{T}
	cholesky_factors::V
	side_length::J    # side length of the grid
end


# This function will sample parameters from the prior, but only keeps those that lead to a valid Choleksy decomposition
function Parameters(K::Integer, ξ)

	# Sample parameters from the prior
	θ = [rand(ϑ, K) for ϑ in ξ.Ω]

	# Covariance parameters associated with the Gaussian process
	λ_idx = findfirst(ξ.parameter_names .== "λ")
	ν_idx = findfirst(ξ.parameter_names .== "ν")
	L_idx = findfirst(ξ.parameter_names .== "L")
	ω_idx = findfirst(ξ.parameter_names .== "ω")

	λ = θ[λ_idx]
	ν = θ[ν_idx]
	L = θ[L_idx]
	ω = θ[ω_idx]


	cholesky_factors = map(1:K) do k
			temp_theta=θ
			bool = 1
			S̃ = mapslices(s -> anisotransform(s, ω[k], L[k]), S, dims = 2)
			D = pairwise(Haversine(), S̃, S̃, dims = 1)/1000 # Apply the Haversine distance to each pair in tranS
			try
			 	A = maternchols(D, λ[k], ν[k])
				A
			catch
				bool = 0

			end
			while bool == 0
					newtheta =	[rand(ϑ, 1) for ϑ in ξ.Ω]

					global θ[λ_idx][k]=newtheta[λ_idx][1]
					global θ[ν_idx][k]=newtheta[ν_idx][1]
					global θ[L_idx][k]=newtheta[L_idx][1]
					global θ[ω_idx][k]=newtheta[ω_idx][1]

					λ = θ[λ_idx]
					ν = θ[ν_idx]
					L = θ[L_idx]
					ω = θ[ω_idx]

					S̃ = mapslices(s -> anisotransform(s, ω[k], L[k]), S, dims = 2)
					D = pairwise(Haversine(), S̃, S̃, dims = 1)/1000 # Apply the Haversine distance to each pair in tranS

					try
						bool = 1
						A = maternchols(D, λ[k], ν[k])
						A
					catch
						bool = 0
					end
			end
			A = A[:, :] # convert from NxNx1 array to NxN matrix

		    A
		end
	# Concatenate into a matrix
	θ = hcat(θ...)'
	θ = Float32.(θ)

	Parameters(θ, cholesky_factors, ξ.side_length)
end


# ---- Process simulation ----

c_τ  = 0  # Censored values will be set to this constant.
# Note that the standard marginal distribution function is taken to be F_* ∼ Pareto(1).
# This can be changed within the following function 'simulate', but is not automatically implemented through a functional argument.
c = 1/(1-τ) #Censoring threshold on Pareto margins

# This function will simulate realistions from an anistropic HW process
function simulate(parameters::Parameters, m::Integer)

	K = size(parameters, 2)
	δ = parameters.θ[2, :]

	Z = Folds.map(1:K) do k
		A = parameters.cholesky_factors[k]
		z = simulategaussianprocess(A, m)
		z = cdf.(Normal(),z)
		z = -log.(1 .- z)
		z = (1-δ[k]) .*  z
		# Raise to power delta
		R =  -log.(1 .-rand(size(z,2)...)) #
		R =  (δ[k]) .* R
		z = z .+ R'
		if δ[k] == 0.5 z = 1 .- exp.(-2 .* z).*(1 .+ 2 .* z) else z = 1 .- (δ[k] ./ (2 .* δ[k]	.- 1)) .* exp.(- z ./ δ[k]) .+ ((1 .- δ[k]) ./ (2*δ[k] .- 1 )) .* exp.(-z ./ (1-δ[k])) end

		#One hot encoding + standard marginal transformation

		z = 1 ./ ( 1 .- z) # F_*^{-1} ∼ Pareto(1). CAN BE CHANGED
		z[z .< c] .= c_τ

		O =  (z .>= c)
		z = Float32.(z)
		z = reshape(z, parameters.side_length, parameters.side_length, 1, :)
		O = Float32.(O)
		O = reshape(O, parameters.side_length, parameters.side_length, 1, :)
		z = cat(z, O, dims=3)
	end

	return Z
end



seed!(1)

#Define outer architecture. Depends on G
if G==4
	ψ = Chain(
		Conv((2, 2), 2 => 64,  relu),
		#BatchNorm(64),
		Conv((2, 2),  64 => 128,  relu),
		Conv((2, 2),  128 => 256, relu),
		Flux.flatten
		)
elseif G==8
	ψ = Chain(
		Conv((5, 5), 2 => 64,  relu),
		#BatchNorm(64),
		Conv((3, 3),  64 => 128,  relu),
		Conv((2, 2),  128 => 256, relu),
		Flux.flatten
		)
elseif G==16
	ψ = Chain(
		Conv((10, 10), 2 => 64,  relu),
		#BatchNorm(64),
		Conv((5, 5),  64 => 128,  relu),
		Conv((3, 3),  128 => 256, relu),
		Flux.flatten
		)
elseif G==24
	ψ = Chain(
	        Conv((9, 9), 2 => 32,  relu),
	      #  BatchNorm(32),
		Conv((10, 10), 32 => 64,  relu),
		Conv((5, 5),  64 => 128,  relu),
		Conv((3, 3),  128 => 256, relu),
		Flux.flatten
		)
elseif G==32
	ψ = Chain(
	        Conv((17, 17), 2 => 32,  relu),
	    #    BatchNorm(32),
		Conv((10, 10), 32 => 64,  relu),
		Conv((5, 5),  64 => 128,  relu),
		Conv((3, 3),  128 => 256, relu),
		Flux.flatten
		)
end

p=5 #Number of parameters

#Define inner architecture
ϕ = Chain(
	Dense(256, 500, relu),
	Dense(500, p)

)

θ̂ = DeepSet(ψ, ϕ)

# ---- Training ----

#Create directory to save model fits during training
int_path = "intermediates/HW"
if !isdir(int_path) mkpath(int_path) end
savepath = "$int_path/runs"


m̃ = [46,138,276] #Use pre-training with m̃

# Train estimator
estimators = trainx(θ̂,  Parameters, simulate,	m̃ , ξ=ξ, K=K,savepath = savepath,
 epochs_per_θ_refresh = 30, epochs_per_Z_refresh=5)

θ̂ = estimators[end] # estimator for m = 276 only

using Flux: loadparams!
loadparams!(θ̂, loadbestweights(string(savepath,"_m",m̃[end]))) # Load best saved estimator


# ---- Assess the estimator ----

parameters = Parameters(1000, ξ) #Draw test parameters and data
Z = simulate(parameters, m̃[end])

#Assess the estimator for the test data
assessment = assess([θ̂ ], parameters,Z)

test_risk=risk(assessment)
print(test_risk)
savepath = "$int_path/estimates"
if !isdir(savepath) mkpath(savepath) end

#Save estimator diagnostics
CSV.write(savepath * "/test_estimates.csv", assessment.df)
CSV.write(savepath * "/runtime.csv", assessment.runtime)
CSV.write(savepath * "/test_risk.csv", test_risk)

#We now assess the estimator for single parameter values.
for it in 1:20

	parameters = Parameters(1, ξ)
	Z = [simulate(parameters, m̃[end])[1] for i in 1:1000]

    test_assessment = assess([θ̂ ], parameters, Z)

	CSV.write(savepath * string("/single_par_test_estimates", it, ".csv"), test_assessment.df)
end


# ---- Apply the estimator ----

#Load in application data
data=load(string("data/saudiPM25_G",G,".Rdata"))

using Flux: gpu
using NeuralEstimators: _runondevice
θ̂ = gpu(θ̂) #Put estimator on GPU

# We convert the input data to an optimised data type
input = broadcast.(Float32, data["Z"])
input = gpu(input)

# The function _runondevice will allow batch estimation if input data is too large for GPUs to handle
data_est=_runondevice(θ̂, input, true)

import Tables: table
CSV.write(savepath * string("/application_estimates.csv"), Tables.table(data_est))
