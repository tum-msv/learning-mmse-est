type DiscreteMMSE
    rho # snr

    W
    bias
    
    DiscreteMMSE() = new()
end

function DiscreteMMSE(snr,get_cov;nSamples=1)
    est = DiscreteMMSE()
    est.rho = 10^(0.1*snr)

    est.W = Array{Any}(nSamples)
    est.bias = Array{Any}(nSamples)

    for i in 1:nSamples
        C = get_cov() # get random sample from cov. prior
        nAntennas   = size(C,1)
        est.W[i]    = C/(C + 1/est.rho * eye(nAntennas)) # MMSE filter at sample i
        est.bias[i] = real(logdet(eye(nAntennas) - est.W[i])) # bias term at sample i
    end

    return est
end


"""
    estimate(est::DiscreteMMSE, y)

For each batch `b`, estimate channels `h[:,t,b]` from
observations `y[:,1], ..., y[:,T,b]` by formula

h[:,t,b] = W_est[b] * y[:,t,b]

with

W_est[b] = sum_i{ weights_i[b] W_i }

and

weights_i[b] = softmax( rho*sum_t{ y[:,t,b]' W[i] * y[:,t,b] } + nCoherence*bias[i] )
"""
function estimate(est::DiscreteMMSE, y)
    nAntennas  = size(y,1)
    nCoherence = size(y,2)
    nBatches   = size(y,3);
    nSamples   = length(est.W);

    # calculate weights for all batches
    exps = zeros(nSamples,nBatches)
    for b in 1:nBatches, t in 1:nCoherence, i in 1:nSamples
        exps[i,b] += real(y[:,t,b]'*est.W[i]*y[:,t,b])[1] * est.rho + est.bias[i]
    end
    exps = exps .- maximum(exps,1)

    weights = exp.(exps)
    weights = weights./sum(weights,1)


    # Calculate estimated MMSE filters and channel estimates
    hest = zeros(nAntennas, nCoherence, nBatches) + 1im*zeros(nAntennas, nCoherence, nBatches)
    for b = 1:nBatches
        W_est = zeros(nAntennas, nAntennas)
        for i=1:nSamples
            W_est += est.W[i]*weights[i,b];
        end
        hest[:,:,b] = W_est*y[:,:,b]
    end

    return hest
end
