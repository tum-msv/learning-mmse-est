#
# Each time this file is executed, two lines will be appended to
# 'results/figure4.csv'. Run 50 times to generate a boxplot similar
# to the one in
#
#   D. Neumann, T. Wiese, and W. Utschick, Learning the MMSE Channel Estimator,
#   IEEE Transactions on Signal Processing, 2018.
#
# (Closing Julia in between runs will ensure that the TensorFlow graph is
# reset and that memory is freed up after each run.)
#

push!(LOAD_PATH,".")
using DataFrames
using CSV
import SCM3GPP; const scm = SCM3GPP
import CondNormalMMSE; const mmse = CondNormalMMSE
import CondNormalTF; const cntf = CondNormalTF

include("sim_helpers.jl")
include("more_estimators.jl") # OMP and Genie MMSE

verbose = true
#-------------------------------------
# Simulation parameters
#
write_file = true
filename   = "results/figure4.csv"
nBatches   = 100
nBatchSize = 100

#-------------------------------------
# Channel Model
#
snr        = 0 # [dB]
AS         = 2.0 # standard deviation of Laplacian (angular spread)
nCoherence = 1
Channel    = scm.SCMMulti(pathAS=AS, nPaths=3)
# method that generates "nBatch" channel realizations
get_channel(nAntennas, nCoherence, nBatches) = scm.generate_channel(Channel, nAntennas, nCoherence=nCoherence, nBatches = nBatches)
# method that samples C_delta from delta prior
get_cov(nAntennas) = scm.toeplitzHe( scm.generate_channel(Channel, nAntennas, nCoherence=1)[2][:] )
# get circulant vector that generates all covariance matrices for arbitrary delta (here: use delta=0)
get_circ_cov_generator(nAntennas) = real(scm.best_circulant_approximation(scm.scm_channel([0.0],[1.0],nAntennas,AS=AS)[2]))

#-------------------------------------
# Learning Algorithm parameters
#
nLayers = 2
nLearningBatches   = 6000
nLearningBatchSize = 20

init_params = Vector{Any}(2)

init_params[1] = Dict{Symbol,Any}()
init_params[1][:nAntennas]   = [   8;  16;  32;  64]
init_params[1][:nBatches]    = [1000;1000;1000]
init_params[1][:nBatchSize]  = [  20;  20;  20]
init_params[1][:snrs]        = [ snr; snr; snr]
init_params[1][:get_channel] = (nAntennas,nBatches) -> get_channel(nAntennas, nCoherence, nBatches)
init_params[1][:learning_rates] = Dict{Symbol,Vector{Float64}}()
init_params[1][:learning_rates][:CircReLUHier] = 1e-4*64./init_params[1][:nAntennas]

init_params[2] = Dict{Symbol,Any}()
init_params[2][:nAntennas]   = [   8;  16;  32;  64; 128]
init_params[2][:nBatches]    = [1000;1000;1000;2000]
init_params[2][:nBatchSize]  = [  20;  20;  20;  20]
init_params[2][:snrs]        = [ snr; snr; snr; snr]
init_params[2][:get_channel] = (nAntennas,nBatches) -> get_channel(nAntennas, nCoherence, nBatches)
init_params[2][:learning_rates] = Dict{Symbol,Vector{Float64}}()
init_params[2][:learning_rates][:CircReLUHier] = 1e-4*64./init_params[2][:nAntennas]

df = DataFrame()
# read results from previous run
if isfile(filename)
    df = CSV.read(filename)
end

srand(size(df,1)) # use number of entries in results as seed for random number generator

results = DataFrame()
for i in 1:length(init_params)
    nAntennas = init_params[i][:nAntennas][end]

    # Network estimators
    nn_est = Dict{Symbol,Any}()
    nn_est[:CircReLU] = cntf.ConvNN(nLayers, init_params[i][:nAntennas][end],
                                    transform = circ_trans,
                                    learning_rate = init_params[i][:learning_rates][:CircReLUHier][end])
    nn_est_hier = Dict{Symbol,Any}()
    nn_est_hier[:CircReLUHier] = cntf.ConvNN(nLayers, init_params[i][:nAntennas][1],
                                             transform = circ_trans,
                                             learning_rate = init_params[i][:learning_rates][:CircReLUHier][1])

    # Hierarchical training:
    init_hier!(nn_est_hier, init_params[i], verbose = verbose)
    train!(nn_est_hier, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize), verbose = verbose)

    train!(nn_est, snr = snr, nBatches = nLearningBatches + sum(init_params[i][:nBatches]), get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize), verbose = verbose)    
    
    algs = Dict{Symbol,Any}()
    for (alg,nn) in nn_est
        algs[alg] = (y,h,h_cov) -> cntf.estimate(nn, y)
    end
    for (alg,nn) in nn_est_hier
        algs[alg] = (y,h,h_cov) -> cntf.estimate(nn, y)
    end

    (errs,rates) = evaluate(algs, snr = snr, nBatches = nBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nBatchSize), verbose = verbose)

    for alg in keys(algs)
        new_row = DataFrame(MSE        = errs[alg],
                            rate       = rates[alg],
                            Algorithm  = alg,
                            SNR        = snr, 
                            nAntennas  = nAntennas,
                            nCoherence = nCoherence)

        if isempty(results)
            results = new_row
        else
            results = vcat(results,new_row)
        end
    end
end
@show results
if write_file
    CSV.write(filename, results, append=true)
end
