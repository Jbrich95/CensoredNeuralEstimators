
using NeuralEstimators
using Flux
using CSV
using DataFrames
using Random: seed!
using RData


K=1000 #In paper, K = 125000


data=load(string("training_replicates/IMSP_reps.Rdata")) #Load pre-simulated replicates


# ---- Architecture ----

seed!(1)

#Define outer architecture.
ψ = Chain(
	Conv((10, 10), 2 => 64,  relu),
	Conv((5, 5),  64 => 128,  relu),
	Conv((3, 3),  128 => 256, relu),
	Flux.flatten
	)

p=2 #Number of parameters

#Define inner architecture
ϕ = Chain(
	Dense(256+1, 500, relu), #If tau not include, then use Dense(256, 500, relu)
	Dense(500, p)
)

θ̂ = DeepSet(ψ, ϕ)



# ---- Training ----



#Create directory to save model fits during training
int_path = "intermediates/IMSP_random_tau"
if !isdir(int_path) mkpath(int_path) end
savepath = "$int_path/runs"

#Reformat tau_vectors to be Vector{Vector}
tau_train= [[vec(data["tau_train"])[i]] for i ∈ eachindex(data["Z_train"])]
tau_val= [[vec(data["tau_val"])[i]] for i ∈ eachindex(data["Z_val"])]
tau_test= [[vec(data["tau_test"])[i]] for i ∈ eachindex(data["Z_test"])]

m̃ = [10,50,100,200] #Use pre-training with m̃

# Traing estimator. Note that to include tau_train/tau_val in the architecture, we pass the replicates as a tuple, e.g., (data["Z_train"],tau_train)
estimators = trainx(θ̂, data["theta_train"], data["theta_val"], (data["Z_train"],tau_train), (data["Z_val"],tau_val),
					m̃, savepath = savepath)

#If you do not want to include tau, then run instead:
# estimators = trainx(θ̂, data["theta_train"], data["theta_val"], data["Z_train"], data["Z_val"], m̃, savepath = savepath)

θ̂ = estimators[end] # extract estimator for m = 200 only

using Flux: loadparams!
loadparams!(θ̂, loadbestweights(string(savepath,"_m",m̃[end]))) # Load best saved estimator


# ---- Assess the estimator ----


#Assess the estimator for the test data
assessment = assess([θ̂], data["theta_test"], (data["Z_test"],tau_test) )
test_risk=risk(assessment)
print(test_risk)

#Save estimator diagnostics
savepath = "$int_path/estimates"
if !isdir(savepath) mkpath(savepath) end
CSV.write(savepath * "/test_estimates.csv", assessment.df)
CSV.write(savepath * "/runtime.csv", assessment.runtime)
CSV.write(savepath * "/test_risk.csv", test_risk)



using Flux: gpu
using NeuralEstimators: _runondevice
θ̂ = gpu(θ̂) #Put estimator on GPU

#We now assess the estimator for single parameter values.
for it in 1:10
	test_input=load(string("training_replicates/IMSP_test_reps",it,".Rdata"))
	tau_test= [[vec(test_input["tau_test"])[1]] for i ∈ eachindex(test_input["Z_test"])]
	test_assessment = assess([θ̂], test_input["theta_test"], (test_input["Z_test"],tau_test) )

	CSV.write(savepath * string("/single_par_test_estimates", it, ".csv"), test_assessment.df)
end
