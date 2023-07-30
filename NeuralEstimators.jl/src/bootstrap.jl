"""
	interval(θ̃, θ̂ = nothing; type::String, probs = [0.05, 0.95], parameter_names)

Compute a confidence interval using the p × B matrix of bootstrap samples, `θ̃`,
where p is the number of parameters in the model.

If `type = "quantile"`, the interval is constructed by simply taking the quantiles of
`θ̃`, and if `type = "reverse-quantile"`, the so-called
[reverse-quantile](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Methods_for_bootstrap_confidence_intervals)
method is used. In both cases, the quantile levels are controlled by the argument `probs`.

The rows can be named with a vector of strings `parameter_names`.

The return type is a p × 2 matrix, whose first and second columns respectively
contain the lower and upper bounds of the interval.

# Examples
```
using NeuralEstimators
p = 3
B = 50
θ̃ = rand(p, B)
θ̂ = rand(p)
interval(θ̃)
interval(θ̃, θ̂, type = "basic")
```
"""
function interval(θ̃, θ̂ = nothing; type::String = "percentile", probs = [0.05, 0.95], parameter_names = ["θ$i" for i ∈ 1:size(θ̃, 1)])

	#TODO add assertions and add type on θ̃
	p, B = size(θ̃)
	type = lowercase(type)

	if type ∈ ["percentile", "quantile"]
		ci = mapslices(x -> quantile(x, probs), θ̃, dims = 2)
	elseif type ∈ ["basic", "reverse-percentile", "reverse-quantile"]
		isnothing(θ̂) && error("`θ̂` must be provided if `type` is 'basic', 'reverse-percentile', or 'reverse-quantile'")
		q = mapslices(x -> quantile(x, probs), θ̃, dims = 2)
		ci = [[2θ̂[i] - q[i, 2], 2θ̂[i] - q[i, 1]] for i ∈ 1:p]
		ci = hcat(ci...)'
	else
		error("argument `type` not matched: it should be one of 'percentile', 'basic', 'studentised', or 'bca'.")
	end

	# Add labels to the confidence intervals
	l = ci[:, 1]
	u = ci[:, 2]
	labelinterval(l, u, parameter_names)
end

function interval(estimator::IntervalEstimator, Z; parameter_names = nothing, use_gpu::Bool = true)

	ci = _runondevice(estimator, Z, use_gpu)
	ci = cpu(ci)

	@assert size(ci, 1) % 2 == 0
	p = size(ci, 1) ÷ 2
	if isnothing(parameter_names)
		parameter_names = ["θ$i" for i ∈ 1:p]
	else
		@assert length(parameter_names) == p
	end
	labelinterval(ci, parameter_names)
end


function labelinterval(l::V, u::V, parameter_names = ["θ$i" for i ∈ length(l)]) where V <: AbstractVector
	@assert length(l) == length(u)
	NamedArray(hcat(l, u), (parameter_names, ["lower", "upper"]))
end

function labelinterval(ci::V, parameter_names = ["θ$i" for i ∈ (length(ci) ÷ 2)]) where V <: AbstractVector

	@assert length(ci) % 2 == 0
	p = length(ci) ÷ 2
	l = ci[1:p]
	u = ci[(p+1):end]
	labelinterval(l, u, parameter_names)
end

function labelinterval(ci::M, parameter_names = ["θ$i" for i ∈ (size(ci, 1) ÷ 2)]) where M <: AbstractMatrix

	@assert size(ci, 1) % 2 == 0
	p = size(ci, 1) ÷ 2
	K = size(ci, 2)

	map(1:K) do k
		lₖ = ci[1:p, k]
		uₖ = ci[(p+1):end, k]
		labelinterval(lₖ, uₖ, parameter_names)
	end
end

# ---- Parameteric bootstrap ----

"""
	bootstrap(θ̂, parameters::P, Z) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(θ̂, parameters::P, simulator, m::Integer; B = 400) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(θ̂, Z; B = 400, blocks = nothing)

Generates `B` bootstrap estimates from an estimator `θ̂`.

Parametric bootstrapping is facilitated by passing a single parameter
configuration, `parameters`, and corresponding simulated data, `Z`, whose length
implicitly defines `B`. Alternatively, one may provide a `simulator` and the
desired sample size, in which case the data will be simulated using
`simulator(parameters, m)`.

Non-parametric bootstrapping is facilitated by passing a single data set, `Z`.
The argument `blocks` caters for block bootstrapping, and it should be a vector
of integers specifying the block for each replicate. For example, with 5 replicates,
the first two corresponding to block 1 and the remaining three corresponding to
block 2, `blocks` should be `[1, 1, 2, 2, 2]`. The resampling algorithm aims to
produce resampled data sets that are of a similar size to `Z`, but this can only
be achieved exactly if all blocks are equal in length.

The keyword argument `use_gpu` is a flag determining whether to use the GPU,
if it is available (default `true`).

The return type is a p × `B` matrix, where p is the number of parameters in the model.
"""
function bootstrap(θ̂, parameters::P, simulator, m::Integer; B::Integer = 400, use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"

	# simulate the data
	v = [simulator(parameters, m) for i ∈ 1:B]
	if typeof(v[1]) <: Tuple
		z = vcat([v[i][1] for i ∈ eachindex(v)]...)
		x = vcat([v[i][2] for i ∈ eachindex(v)]...)
		v = (z, x)
	else
		v = vcat(v...)
	end
	
	θ̃ = _runondevice(θ̂, v, use_gpu)
	return θ̃
end

function bootstrap(θ̂, parameters::P, Z̃; use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"
	θ̃ = _runondevice(θ̂, Z̃, use_gpu)
	return θ̃
end


# ---- Non-parametric bootstrapping ----

function bootstrap(θ̂, Z; B::Integer = 400, use_gpu::Bool = true, blocks = nothing)

	# Generate B bootstrap samples of Z
	if !isnothing(blocks)
		Z̃ = _blockresample(Z, B, blocks)
	else
		m = numberreplicates(Z)
		Z̃ = [subsetdata(Z, rand(1:m, m)) for _ in 1:B]
	end
	# Estimate the parameters for each bootstrap sample
	θ̃ = _runondevice(θ̂, Z̃, use_gpu)

	return θ̃
end

# simple wrapper to handle the common case that the user forgot to extract the
# array from the single-element vector returned by a simulator
function bootstrap(θ̂, Z::V; args...) where {V <: AbstractVector{A}} where A

	@assert length(Z) == 1
	Z = Z[1]
	return bootstrap(θ̂, Z; args...)
end

"""
Generates `B` bootstrap samples by sampling `Z` with replacement, with the
replicates grouped together in `blocks`, integer vector specifying the block for
each replicate.

For example, with 5 replicates, the first two corresponding to block 1 and the
remaining three corresponding to block 2, `blocks` should be `[1, 1, 2, 2, 2]`.

The resampling algorithm aims to produce data sets that are of a similar size to
`Z`, but this can only be achieved exactly if the blocks are of equal size.
"""
function _blockresample(Z, B::Integer, blocks)

	@assert length(blocks) == numberreplicates(Z) "The number of replicates and the length of `blocks` must match: we recieved `numberreplicates(Z) = $(numberreplicates(Z))` and `length(blocks) = $(length(blocks))`"

	m = length(blocks)
	unique_blocks = unique(blocks)
	num_blocks    = length(unique_blocks)

	# Define c ≡ median(block_counts)/2 and d ≡ maximum(block_counts).
	# The following method ensures that m̃ ∈ [m - c, m - c + d), where
	# m is the sample size  (with respect to the number of independent replicates)
	# and m̃ is the sample size of the resampled data set.

	block_counts = [count(x -> x == i, blocks) for i ∈ unique_blocks]
	c = median(block_counts) / 2
	Z̃ = map(1:B) do _
		sampled_blocks = Int[]
		m̃ = 0
		while m̃ < m - c
			push!(sampled_blocks, rand(unique_blocks))
			m̃ += block_counts[sampled_blocks[end]]
		end
		idx = vcat([findall(x -> x == i, blocks) for i ∈ sampled_blocks]...)
		subsetdata(Z, idx)
	end

	return Z̃
end
