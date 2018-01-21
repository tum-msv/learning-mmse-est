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
nBatches   = 50
nBatchSize = 100

#-------------------------------------
# Channel Model
#
snr        = 0 # [dB]
rho        = 10^(0.1*snr);
antennas   = [8,16,32,64,96,128]
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
learning_rates = 2e-4*128./antennas # make learning rates dependend on nAntennas
nLayers        = 2
nLearningBatches   = 8000
nLearningBatchSize = 20
# cnn_use_resize: networks with more antennas are initialized from networks with
#   less antennas (only effective if cnn_load_and_save is false or no file found)
cnn_use_resize    = true
# save network variable values
cnn_load_and_save = false
cnn_filename(alg,nAntennas) = @sprintf "results/%s_nPaths%.0f_nCoherence%.0f_AS%.1f_snr%.0f_nAntennas%.0f.jld" alg nPaths nCoherence AS snr nAntennas



results = DataFrame()
algs       = Dict{Symbol,Any}()
nn_est     = Dict{Symbol,Any}()
nn_files   = Dict{Symbol,String}()
cn_est     = Dict{Symbol,Any}()
for iAntenna in 1:length(antennas)
    nAntennas     = antennas[iAntenna]
    learning_rate = learning_rates[iAntenna]

    verbose && println("Simulating with ", nAntennas, " antennas")

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
    if iAntenna == 1 || !cnn_use_resize
        nn_est[:CircReLU] = cntf.ConvNN(nLayers, nAntennas,  transform = circ_trans, learning_rate = learning_rate)
        nn_est[:ToepReLU] = cntf.ConvNN(nLayers, 2nAntennas, transform = toep_trans, learning_rate = learning_rate)

        # Initialize CircSoftmax NN with kernel and bias from Fast MMSE
        c = get_circ_cov_generator(nAntennas)
        w = c./(c .+ (1/rho))
        bias    = sum(log.(1 .- w)) * nCoherence
        biases  = [bias * ones(nAntennas), zeros(nAntennas)]
        kernels = [w*nCoherence*rho, w[end:-1:1]]
        nn_est[:CircSoftmax] = cntf.ConvNN(kernels, biases,  transform = circ_trans, activation = cntf.nn.softmax, learning_rate = learning_rate)
        nn_est[:ToepSoftmax] = cntf.ConvNN(nLayers, 2nAntennas, transform = toep_trans, activation = cntf.nn.softmax, learning_rate = learning_rate)
    elseif cnn_use_resize
        nn_est[:CircReLU] = cntf.resize(nn_est[:CircReLU], nAntennas, learning_rate = learning_rate)
        nn_est[:ToepReLU] = cntf.resize(nn_est[:ToepReLU], 2nAntennas, learning_rate = learning_rate)
        nn_est[:CircSoftmax] = cntf.resize(nn_est[:CircSoftmax], nAntennas, learning_rate = learning_rate)
        nn_est[:ToepSoftmax] = cntf.resize(nn_est[:ToepSoftmax], 2nAntennas, learning_rate = learning_rate)
    end

    if cnn_load_and_save
        nn_files[:CircReLU] = cnn_filename("CircReLU",  nAntennas)
        nn_files[:ToepReLU] = cnn_filename("ToepReLU", nAntennas)
        nn_files[:CircSoftmax] = cnn_filename("CircSoftmax",  nAntennas)
        nn_files[:ToepSoftmax] = cnn_filename("ToepSoftmax", nAntennas)
        for (alg,nn) in nn_est
            cntf.load!(nn, nn_files[alg])
        end
    end
    train!(nn_est, snr = snr, nBatches = nLearningBatches, get_channel = () -> get_channel(nAntennas, nCoherence, nLearningBatchSize), verbose = verbose)
    if cnn_load_and_save
        for (alg,nn) in nn_est
            cntf.save(nn, nn_files[alg])
        end
    end
    for (alg,nn) in nn_est
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


# function plot_results()
#     for alg in keys(algs)
#         plot(antennas, results[results[:Algorithm] .== alg, :MSE], label=alg)
#     end
#     legend()
# end
