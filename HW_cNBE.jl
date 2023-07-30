using NeuralEstimators
using Distances: pairwise, Euclidean, haversine
using Folds
using LinearAlgebra
using Distributions
import NeuralEstimators: simulate # not necessary to overlaod this function, I just do it because it grants access to a few default methods
import Distances: Haversine

using Flux
using CSV
using Tables
using Random: seed!
using RData

print(ARGS)
type=ARGS[1]
censor_p=ARGS[2]
J=ARGS[3]

m_max=ARGS[4]
G=ARGS[5]
batch_size=ARGS[6]

print(type)
print(censor_p)
print(J)
print(m_max)

# Set the prior distribution
Ω = (
	λ = Uniform(20.0, 500.0),
	δ = Uniform(0.0, 1.0),
	ν = Uniform(0.1, 4.0),
	L = Uniform(0.5,3.5),
	ω = Uniform(-pi/2,0.0)
)

# Set your spatial domain

S = expandgrid(LinRange(45.05,45.05+(parse(Int,G)-1)*0.1, parse(Int,G)), LinRange(23.75,23.75+(parse(Int,G)-1)*0.1, parse(Int,G)))
D = pairwise(Haversine(6378.388), S,S , dims = 1)

censor_thresh=1/(1-parse(Float32,censor_p))
# Tuple to store the important model information
ξ = (
	Ω = Ω,
	S = S,
	p = length(Ω),
	d = size(S, 1),
	parameter_names = String.(collect(keys(Ω))),
	side_length = parse(Int,G)
)

# ---- Sampling parameters ----

struct Parameters{T, V, J} <: ParameterConfigurations
	θ::Matrix{T}
	cholesky_factors::V
	side_length::J    # side length of the grid
end

function anisotransform(s, ω, L)
  Lmat = Diagonal([1.0, 1/L])
  ψmat = [cos(ω) -sin(ω); sin(ω) cos(ω)] # this notation fills a matrix by row: see [1 2; 3 4]
  Lmat * ψmat * s
end


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


# ---- Marginal simulation ----


function simulate(parameters::Parameters, m::Integer)

	K = size(parameters, 2)
	δ = parameters.θ[2, :]

	Z = Folds.map(1:K) do k
		A = parameters.cholesky_factors[k]
		z = simulategaussianprocess(A, m)
		z = cdf.(Normal(),z)
		z = 1 ./ ( 1 .- z)
		z = (1-δ[k]) .* log.( z )
		# Raise to power delta
		R =  (1 ./ (1 .-rand(size(z,2)...)))
		R =  (δ[k]) .* log.( R )
		z = z .+ R'
		if δ[k] == 0.5 z = 1 .- exp.(-2 .* z).*(1 .+ 2 .* z) else z = 1 .- (δ[k] ./ (2 .* δ[k]	.- 1)) .* exp.(- z ./ δ[k]) .+ ((1 .- δ[k]) ./ (2*δ[k] .- 1 )) .* exp.(-z ./ (1-δ[k])) end
		#One hot encoding + pareto transformation

		z = 1 ./ ( 1 .- z)
		z[z .< censor_thresh] .= 0

		O =  (z .>= censor_thresh)
		z = Float32.(z)
		z = reshape(z, parameters.side_length, parameters.side_length, 1, :)
		O = Float32.(O)
		O = reshape(O, parameters.side_length, parameters.side_length, 1, :)
		z = cat(z, O, dims=3)
	end

	return Z
end





if G=="4"
	ψ = Chain(
		Conv((2, 2), 2 => 64,  relu),
		#BatchNorm(64),
		Conv((2, 2),  64 => 128,  relu),
		Conv((2, 2),  128 => 256, relu),
		Flux.flatten
		)
elseif G=="8"
	ψ = Chain(
		Conv((5, 5), 2 => 64,  relu),
		#BatchNorm(64),
		Conv((3, 3),  64 => 128,  relu),
		Conv((2, 2),  128 => 256, relu),
		Flux.flatten
		)
elseif G=="16"
	ψ = Chain(
		Conv((10, 10), 2 => 64,  relu),
		#BatchNorm(64),
		Conv((5, 5),  64 => 128,  relu),
		Conv((3, 3),  128 => 256, relu),
		Flux.flatten
		)
elseif G=="24"
	ψ = Chain(
	        Conv((9, 9), 2 => 32,  relu),
	      #  BatchNorm(32),
		Conv((10, 10), 32 => 64,  relu),
		Conv((5, 5),  64 => 128,  relu),
		Conv((3, 3),  128 => 256, relu),
		Flux.flatten
		)
elseif G=="32"
	ψ = Chain(
	        Conv((17, 17), 2 => 32,  relu),
	    #    BatchNorm(32),
		Conv((10, 10), 32 => 64,  relu),
		Conv((5, 5),  64 => 128,  relu),
		Conv((3, 3),  128 => 256, relu),
		Flux.flatten
		)
end

p=5

ϕ = Chain(
	Dense(256, 500, relu),
	Dense(500, p)



)

θ̂ = DeepSet(ψ, ϕ)

# ---- Training ----

 model = string("HW_type",  type,"_m",m_max,"_p",censor_p,"_N",J,"_G",G)
m = [46,138,parse(Int,m_max)]

int_path = "intermediates/$model"
if !isdir(int_path) mkpath(int_path) end
savepath = "$int_path/runs"



estimators = trainx(θ̂,  Parameters, simulate,	m, ξ=ξ, K=parse(Int,J),savepath = savepath, epochs_per_θ_refresh = 10, batchsize=parse(Int,batch_size))

estimator = estimators[end] # estimator for m = m_max)-

using Flux: loadparams!
loadparams!(θ̂, loadbestweights(string(savepath,"_m",m_max)))
loadparams!(estimator, loadbestweights(string(savepath,"_m",m_max)))
# ---- Assess the estimator ----


parameters = Parameters(1000, ξ)

m = 276 # number of replicates for each parameter configuration
Z = simulate(parameters, m)


assessment = assess([estimator], parameters,Z)
test_risk=risk(assessment)
print(test_risk)
savepath = "$int_path/estimates"
if !isdir(savepath) mkpath(savepath) end

CSV.write(savepath * "/estimates.csv", assessment.df)
CSV.write(savepath * "/runtime.csv", assessment.runtime)
CSV.write(savepath * "/test_risk.csv", test_risk)

import Tables: table


test_input=load(string("../data/saudiPM25_test_G",G,".Rdata"))

test_est= θ̂(convert(Vector{Array{Float32}}, test_input["Z"]))


CSV.write(savepath * string("/test_estimates.csv"), Tables.table(test_est))


for it in 1:20

	parameters = Parameters(1, ξ)
	Z = [simulate(parameters, m)[1] for i in 1:1000]

    test_assessment = assess([estimator], parameters, Z)

	CSV.write(savepath * string("/test_estimates", it, ".csv"), test_assessment.df)
end
