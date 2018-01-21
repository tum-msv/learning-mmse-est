type MLEst
    rho
    transform
    
    MLEst() = new()
end

function MLEst(rho; transform = (x,_) -> x)
    est = MLEst()
    est.rho = rho
    est.transform = transform
    est
end


function estimate(est::MLEst, y)
    z = est.transform(y, :notransp)
    (nAntennas,nCoherence,nBatches) = size(z)# input data in transformed domain

    cest = max.( 0, sum(abs2,z,2) - nCoherence/est.rho )
    cest = reshape(cest,nAntennas,nBatches)

    W = cest./(cest .+ nCoherence/est.rho)
    W = reshape(W,nAntennas,1,nBatches)

    hest = est.transform( W .* z, :transp)
end
