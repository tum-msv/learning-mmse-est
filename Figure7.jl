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
filename   = "results/figure7.csv"
nBatches   = 50
nBatchSize = 100

#-------------------------------------
# Channel Model
#
snrs       = -15:5:15 # [dB]
nAntennas  = 64
AS         = 2.0 # standard deviation of Laplacian (angular spread)
nPaths     = 3
nCoherence = 1
Channel    = scm.SCMMulti(pathAS=AS, nPaths=nPaths)
# method that generates "nBatches" channel realizations
get_channel(nAntennas, nCoherence, nBatches) = scm.generate_channel(Channel, nAntennas, nCoherence=nCoherence, nBatches = nBatches)
# method that samples C_delta from delta prior
get_cov(nAntennas) = scm.toeplitzHe( scm.generate_channel(Channel, nAntennas, nCoherence=1)[2][:] )
# get circulant vector that generates all covariance matrices for arbitrary delta (here: use delta=0)
get_circ_cov_generator(nAntennas) = real(scm.best_circulant_approximation(scm.scm_channel([0.0],[1.0],nAntennas,AS=AS)[2]))

#-------------------------------------
# Learning Algorithm parameters
#
nLayers = 2
nLearningBatches   = 2000
nLearningBatchSize = 20

nLearningAntennas = [8;16;32;64]
learning_rates    = 1e-4*nLearningAntennas/64 # make learning rates dependend on nAntennas


results = DataFrame()
algs       = Dict{Symbol,Any}()
nn_est     = Dict{Symbol,Any}()
cn_est     = Dict{Symbol,Any}()
for snr in snrs
    rho        = 10^(0.1*snr);

    verbose && println("Simulating with ", snr, " [dB] SNR")

    # Conditionally normal estimators
    cn_est[:FastMMSE]     = mmse.FastMMSE(snr, get_circ_cov_generator(nAntennas))
    cn_est[:CircMMSE]     = mmse.StructuredMMSE(snr, () -> get_cov(nAntennas), nSamples=16*nAntennas, transform = circ_trans)
    cn_est[:ToepMMSE]     = mmse.StructuredMMSE(snr, () -> get_cov(nAntennas), nSamples=16*nAntennas, transform = toep_trans)
#    cn_est[:DiscreteMMSE] = mmse.DiscreteMMSE(snr,   () -> get_cov(nAntennas), nSamples=16*nAntennas)
    cn_est[:CircML]       = mmse.MLEst(rho, transform = circ_trans)
    for (alg,cn) in cn_est
        algs[alg] = (y,h,h_cov) -> mmse.estimate(cn, y)
    end
    algs[:GenieMMSE] = (y,h,h_cov) -> mmse_genie(y, h_cov, snr)
    algs[:GenieOMP]  = (y,h,h_cov) -> omp_genie(y, h)


    # Network estimators
    nn_est[:CircReLU] = cntf.ConvNN(nLayers, nLearningAntennas[1],  transform = circ_trans, learning_rate = learning_rates[1])
    nn_est[:ToepReLU] = cntf.ConvNN(nLayers, 2nLearningAntennas[1], transform = toep_trans, learning_rate = learning_rates[1])

    # Initialize CircSoftmax NN with kernel and bias from Fast MMSE
    c = get_circ_cov_generator(nLearningAntennas[1])
    w = c./(c .+ (1/rho))
    bias    = sum(log.(1 .- w)) * nCoherence
    biases  = [bias * ones(nLearningAntennas[1]), zeros(nLearningAntennas[1])]
    kernels = [w*nCoherence*rho, w[end:-1:1]]
    nn_est[:CircSoftmax] = cntf.ConvNN(kernels, biases,  transform = circ_trans, activation = cntf.nn.softmax, learning_rate = learning_rates[1])
    nn_est[:ToepSoftmax] = cntf.ConvNN(nLayers, 2nLearningAntennas[1], transform = toep_trans, activation = cntf.nn.softmax, learning_rate = learning_rates[1])

    # Initial training for networks with few antennas
    train!(nn_est, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nLearningAntennas[1], nCoherence, nLearningBatchSize), verbose = verbose)

    # Hierarchical training
    for i in 2:length(nLearningAntennas)
        nn_est[:CircReLU] = cntf.resize(nn_est[:CircReLU], nLearningAntennas[i], learning_rate = learning_rates[i])
        nn_est[:ToepReLU] = cntf.resize(nn_est[:ToepReLU], 2nLearningAntennas[i], learning_rate = learning_rates[i])
        nn_est[:CircSoftmax] = cntf.resize(nn_est[:CircSoftmax], nLearningAntennas[i], learning_rate = learning_rates[i])
        nn_est[:ToepSoftmax] = cntf.resize(nn_est[:ToepSoftmax], 2nLearningAntennas[i], learning_rate = learning_rates[i])
        train!(nn_est, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nLearningAntennas[i], nCoherence, nLearningBatchSize), verbose = verbose)
    end
    
    for (alg,nn) in nn_est
        algs[alg] = (y,h,h_cov) -> cntf.estimate(nn,  y)
    end


    (errs,rates) = evaluate(algs, snr = snr, nBatches = nBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize), verbose = verbose)

    for alg in keys(algs)
        @show alg
        @show log10.(errs[alg])
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

    if write_file
        CSV.write(filename, results)
    end
end


# function plot_results()
#     for alg in keys(algs)
#         plot(antennas, results[results[:Algorithm] .== alg, :MSE], label=alg)
#     end
#     legend()
# end
