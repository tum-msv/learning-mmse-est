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
filename   = "results/figure5.csv"
nBatches   = 50
nBatchSize = 100

#-------------------------------------
# Channel Model
#
snr        = 0 # [dB]
rho        = 10^(0.1*snr);
antennas   = [8,16,32,64,96]
AS         = 2.0 # standard deviation of Laplacian (angular spread)
nPaths     = 1
nCoherence = 1
Channel    = scm.SCMMulti(pathAS=AS, nPaths=nPaths)
# method that generates "nBatches" channel realizations
get_channel(nAntennas, nCoherence, nBatches) = scm.generate_channel(Channel, nAntennas, nCoherence=nCoherence, nBatches = nBatches)
# method that samples C_delta from delta prior
get_cov(nAntennas) = scm.toeplitzHe( scm.generate_channel(Channel, nAntennas, nCoherence=1)[2][:] )
# get circulant vector that generates all covariance matrices for arbitrary delta (here: use delta=0)
get_circ_cov_generator(nAntennas) = real(scm.best_circulant_approximation(scm.scm_channel([0.0],[1.0],nAntennas,AS=AS)[2]))


results = DataFrame()
algs       = Dict{Symbol,Any}()
cn_est     = Dict{Symbol,Any}()
for iAntenna in 1:length(antennas)
    nAntennas     = antennas[iAntenna]

    verbose && println("Simulating with ", nAntennas, " antennas")

    # Conditionally normal estimators
    cn_est[:FastMMSE]     = mmse.FastMMSE(snr, get_circ_cov_generator(nAntennas))
    cn_est[:CircMMSE]     = mmse.StructuredMMSE(snr, () -> get_cov(nAntennas), nSamples=16*nAntennas, transform = circ_trans)
    cn_est[:ToepMMSE]     = mmse.StructuredMMSE(snr, () -> get_cov(nAntennas), nSamples=16*nAntennas, transform = toep_trans)
    cn_est[:DiscreteMMSE] = mmse.DiscreteMMSE(snr,   () -> get_cov(nAntennas), nSamples=16*nAntennas)
    cn_est[:CircML]       = mmse.MLEst(rho, transform = circ_trans)
    for (alg,cn) in cn_est
        algs[alg] = (y,h,h_cov) -> mmse.estimate(cn, y)
    end
    algs[:GenieMMSE] = (y,h,h_cov) -> mmse_genie(y, h_cov, snr)
    algs[:GenieOMP]  = (y,h,h_cov) -> omp_genie(y, h)


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
