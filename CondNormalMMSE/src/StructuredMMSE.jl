type StructuredMMSE
    rho # snr
    transform

    W
    bias

    StructuredMMSE() = new()
end

function StructuredMMSE(snr,get_cov; transform = (x,_) -> x, nSamples=1)
    est = StructuredMMSE()
    est.rho = 10^(0.1*snr)
    est.transform = transform

    nAntennas = size(get_cov(),1)
    
    Q = transform(eye(nAntennas), :notransp)
    A = pinv(abs2.(Q*Q')) # note: output transform must be Q', i.e., correct normalization is needed

    est.W    = zeros(size(Q,1),nSamples)
    est.bias = zeros(nSamples)
    for i in 1:nSamples
        C = get_cov() # get random sample from cov. prior
        W = C/(C + 1/est.rho * eye(nAntennas))
        est.W[:,i]  = A*(real(sum(Q*W .* conj(Q),2)))
        est.bias[i] = real(logdet(eye(nAntennas) - Q'*diagm(est.W[:,i])*Q))
    end
    est
end

function estimate(est::StructuredMMSE, y)
    z = est.transform(y, :notransp)
    (nFilterLength,nCoherence,nBatches) = size(z)

    cest = sum(abs2, z, 2)
    cest = reshape(cest,nFilterLength,nBatches)

    exps = est.W'*cest*est.rho .+ est.bias*nCoherence # note cest sum not mean
    exps = exps .- maximum(exps,1)

    weights = exp.(exps)
    weights = weights./sum(weights,1)

    F = est.W*weights;
    F = reshape(F, nFilterLength, 1, nBatches)

    x = est.transform(F.*z, :transp)
end

