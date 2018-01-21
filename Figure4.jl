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
nBatches   = 50
nBatchSize = 100
nMC        = 50

#-------------------------------------
# Channel Model
#
snr        = 0 # [dB]
rho        = 10^(0.1*snr);
antennas   = [64,128]
AS         = 2.0 # standard deviation of Laplacian (angular spread)
nPaths     = 3
nCoherence = 1
Channel    = scm.SCMMulti(pathAS=AS, nPaths=nPaths)
# method that generates "nBatch" channel realizations
get_channel(nAntennas, nCoherence, nBatches) = scm.generate_channel(Channel, nAntennas, nCoherence=nCoherence, nBatches = nBatches)
# method that samples C_delta from delta prior
get_cov(nAntennas) = scm.toeplitzHe( scm.generate_channel(Channel, nAntennas, nCoherence=1)[2][:] )
# get circulant vector that generates all covariance matrices for arbitrary delta (here: use delta=0)
get_circ_cov_generator(nAntennas) = real(scm.best_circulant_approximation(scm.scm_channel([0.0],[1.0],nAntennas,AS=AS)[2]))

#-------------------------------------
# Learning Algorithm parameters
#
learning_rates = 2e-4*128./antennas # make learning rates dependend on nAntennas
nLayers        = 2
nLearningBatches   = 2000
nLearningBatchSize = 20
nLearningAntennas = [ [8;16;32;64], [8;16;32;64;128] ]
learning_rates    = 1e-4*nLearningAntennas/64 # make learning rates dependend on nAntennas

results = DataFrame()
algs   = Dict{Symbol,Any}()
nn_est = Dict{Symbol,Any}()
nn_est_hier = Dict{Symbol,Any}()

for iMC in 1:nMC
    verbose && println("Simulating Monte Carlo iteration ", iMC, "/", nMC)
    for iAntenna in 1:length(antennas)
        nAntennas     = antennas[iAntenna]


        # Network estimators
        nn_est_hier[:CircReLUHier] = cntf.ConvNN(nLayers, nLearningAntennas[iAntenna][1],  transform = circ_trans, learning_rate = learning_rates[iAntenna][1])
        nn_est_hier[:ToepReLUHier] = cntf.ConvNN(nLayers, 2nLearningAntennas[iAntenna][1], transform = toep_trans, learning_rate = learning_rates[iAntenna][1])

        # Hierarchical training:
        # Initial training for networks with few antennas
        train!(nn_est_hier, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nLearningAntennas[iAntenna][1], nCoherence, nLearningBatchSize), verbose = verbose)
        for i in 2:length(nLearningAntennas[iAntenna])
            nn_est_hier[:CircReLUHier] = cntf.resize(nn_est_hier[:CircReLUHier], nLearningAntennas[iAntenna][i], learning_rate = learning_rates[iAntenna][i])
            nn_est_hier[:ToepReLUHier] = cntf.resize(nn_est_hier[:ToepReLUHier], 2nLearningAntennas[iAntenna][i], learning_rate = learning_rates[iAntenna][i])
            train!(nn_est_hier, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nLearningAntennas[iAntenna][i], nCoherence, nLearningBatchSize), verbose = verbose)
        end

        # Non-hierarchical training (use more batches so as
        # to keep the total number of training iterations constant):
        nn_est[:CircReLU] = cntf.ConvNN(nLayers, nAntennas,  transform = circ_trans, learning_rate = learning_rates[iAntenna][end])
        nn_est[:ToepReLU] = cntf.ConvNN(nLayers, 2nAntennas, transform = toep_trans, learning_rate = learning_rates[iAntenna][end])
        train!(nn_est, snr = snr, nBatches = length(nLearningAntennas[iAntenna])*nLearningBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize), verbose = verbose)

        for (alg,nn) in nn_est
            algs[alg] = (y,h,h_cov) -> cntf.estimate(nn,  y)
        end
        for (alg,nn) in nn_est_hier
            algs[alg] = (y,h,h_cov) -> cntf.estimate(nn,  y)
        end

        (errs,rates) = evaluate(algs, snr = snr, nBatches = nBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nBatchSize), verbose = verbose)

        for alg in keys(algs)
            @show alg
            @show errs[alg]
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
end

# function plot_results()
#     for alg in keys(algs)
#         plot(antennas, results[results[:Algorithm] .== alg, :MSE], label=alg)
#     end
#     legend()
# end
