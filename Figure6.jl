#
# Run this file to generate Figure 6 in
#
#   D. Neumann, T. Wiese, and W. Utschick, Learning the MMSE Channel Estimator,
#   IEEE Transactions on Signal Processing, 2018.
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
filename   = "results/figure6.csv"
nBatches   = 100
nBatchSize = 100

#-------------------------------------
# Channel Model
#
snr        = 0 # [dB]
antennas   = [8,16,32,64,96,128]
AS         = 2.0 # standard deviation of Laplacian (angular spread)
nCoherence = 1
Channel    = scm.SCMMulti(pathAS=AS, nPaths=3)
# method that generates "nBatches" channel realizations
get_channel(nAntennas, nCoherence, nBatches) = scm.generate_channel(Channel, nAntennas, nCoherence=nCoherence, nBatches = nBatches)
# method that samples C_delta from delta prior
get_cov(nAntennas) = scm.toeplitzHe( scm.generate_channel(Channel, nAntennas, nCoherence=1)[2][:] )
# get circulant vector that generates all covariance matrices for arbitrary delta (here: use delta=0)
get_circ_cov_generator(nAntennas) = real(scm.best_circulant_approximation(scm.scm_channel([0.0],[1.0],nAntennas,AS=AS)[2]))

#-------------------------------------
# Learning Algorithm parameters
#
learning_rates_relu    = 1e-4*64./antennas # make learning rates dependend on nAntennas
learning_rates_softmax = 1e-3*ones(antennas)
nLayers = 2
nLearningBatches   = 6000
nLearningBatchSize = 20

results = DataFrame()
# read results from previous run
if isfile(filename)
    results = CSV.read(filename)
end

srand(size(results,1)) # use number of entries in results as seed for random number generator
nn_est = Dict{Symbol,Any}()
for iAntenna in 1:length(antennas)
    nAntennas     = antennas[iAntenna]

    verbose && println("Simulating with ", nAntennas, " antennas")

    # Conditionally normal estimators
    cn_est = Dict{Symbol,Any}()
    cn_est[:FastMMSE] = mmse.FastMMSE(snr, get_circ_cov_generator(nAntennas))
    cn_est[:CircMMSE] = mmse.StructuredMMSE(snr, () -> get_cov(nAntennas), nSamples=16*nAntennas, transform = circ_trans)
    cn_est[:ToepMMSE] = mmse.StructuredMMSE(snr, () -> get_cov(nAntennas), nSamples=16*nAntennas, transform = toep_trans)
    cn_est[:CircML]   = mmse.MLEst(snr, transform = circ_trans)

    # Network estimators
    if iAntenna == 1
        nn_est[:CircReLU] = cntf.ConvNN(nLayers, nAntennas, transform = circ_trans, learning_rate = learning_rates_relu[iAntenna])
        nn_est[:ToepReLU] = cntf.ConvNN(nLayers, nAntennas, transform = toep_trans, learning_rate = learning_rates_relu[iAntenna])
        nn_est[:CircSoftmax] = cntf.ConvNN(nLayers, nAntennas, transform = circ_trans, learning_rate = learning_rates_softmax[iAntenna], activation = cntf.nn.softmax)
        nn_est[:ToepSoftmax] = cntf.ConvNN(nLayers, nAntennas, transform = toep_trans, learning_rate = learning_rates_softmax[iAntenna], activation = cntf.nn.softmax)
    else
        nn_est[:CircReLU]    = cntf.resize(nn_est[:CircReLU],    nAntennas, learning_rate = learning_rates_relu[iAntenna])
        nn_est[:ToepReLU]    = cntf.resize(nn_est[:ToepReLU],    nAntennas, learning_rate = learning_rates_relu[iAntenna])
        nn_est[:CircSoftmax] = cntf.resize(nn_est[:CircSoftmax], nAntennas, learning_rate = learning_rates_softmax[iAntenna])
        nn_est[:ToepSoftmax] = cntf.resize(nn_est[:ToepSoftmax], nAntennas, learning_rate = learning_rates_softmax[iAntenna])
    end

    train!(nn_est, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize), verbose = verbose)

    algs = Dict{Symbol,Any}()
    algs[:GenieMMSE] = (y,h,h_cov) -> mmse_genie(y, h_cov, snr)
    algs[:GenieOMP]  = (y,h,h_cov) -> omp_genie(y, h)
    for (alg,cn) in cn_est
        algs[alg] = (y,h,h_cov) -> mmse.estimate(cn, y)
    end
    for (alg,nn) in nn_est
        algs[alg] = (y,h,h_cov) -> cntf.estimate(nn, y)
    end

    (errs,rates) = evaluate(algs, snr = snr, nBatches = nBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nBatchSize), verbose = verbose)

    for alg in keys(algs)
        new_row = DataFrame(MSE        = errs[alg],
                            rate       = rates[alg],
                            Algorithm  = String(alg),
                            SNR        = snr, 
                            nAntennas  = nAntennas,
                            nCoherence = nCoherence)

        if isempty(results)
            results = new_row
        else
            results = vcat(results,new_row)
        end
    end

    if write_file
        CSV.write(filename, results)
    end
    @show results
end
