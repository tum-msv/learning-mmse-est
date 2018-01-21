type FastMMSE
    rho

    v
    bias

    FastMMSE() = new()
end

function FastMMSE(snr,c)
    est = FastMMSE()
    est.rho = 10^(0.1*snr)

    w = c./(c .+ (1/est.rho));
    est.v = fft(w)
    est.bias = sum(log.(1 .- w))

    return est
end

function estimate(est::FastMMSE, y)
    nAntennas  = size(y,1)
    nCoherence = size(y,2)
    nBatches   = size(y,3)

    y = fft(y,1)./sqrt(nAntennas)

    cest = mean(abs.(y).^2,2)
    cest = reshape(cest,nAntennas,nBatches)

    exps = real(ifft(conj(est.v).*fft(cest,1),1))*est.rho*nCoherence .+ est.bias*nCoherence
    exps = exps .- maximum(exps,1)

    weights = exp.(exps)
    weights = weights./sum(weights,1)

    F = real(ifft(est.v .* fft(weights,1),1));
    F = reshape(F,nAntennas,1,nBatches)

    x = y .* F;

    x = ifft(x,1).*sqrt(nAntennas);
end

