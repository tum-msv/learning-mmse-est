# Genie-aided MMSE filter uses true covariance matrix
mmse_genie(y,h_cov,snr) = begin
    rho = 10^(0.1*snr);
    hest = zeros(y)
    (nAntennas,nCoherence,nBatches) = size(y)
    for b in 1:nBatches
        C = scm.toeplitzHe(h_cov[:,b]) # get full cov matrix
        Cr = C + eye(nAntennas)./rho
        hest[:,:,b] = C*(Cr\y[:,:,b])
    end
    hest
end

# Genie-aided OMP algorithm that stops at optimal order (here: optimal wrt. rate)
omp_genie(y,h) = begin
    (nAntennas,nCoherence,nBatches) = size(y)
    nGrid = 4*nAntennas # four times oversampled DFT
    grid = linspace(-1,1, nGrid+1)[1:nGrid]
    A = 1/sqrt(nAntennas) * exp.(1im*pi*(0:nAntennas-1) * grid')
    hest = zeros(h)
    for b in 1:nBatches
        hest[:,:,b] = omp_genie_alg(A,y[:,:,b],h[:,:,b])[1]
    end
    hest
end

"""
    omp_genie_alg(A::Matrix, y::VecOrMat, x0::VecOrMat[, B=A', find_supp = x -> L, solve_restricted = L -> x]) -> x, L
Genie aided Orthogonal Matching Pursuit with optimal support size (based on angle criterion in image space)

# Arguments
* `A`: Full sensing matrix
* `y`: Observation vector or matrix
* `x`: True signal vector or matrix
* `B`: Mismatched filter matrix (e.g., `A*`)
* `find_supp`: function that returns a logical vector with the best index of the input
* `solve_restricted`: function that solves `y=Ax` on with support of `x` restricted to current index set `L`

# Output
* `x`: k-sparse solution the linear system `y=Ax`
* `L`: logical vector of nonzero indexes of `x`
"""
function omp_genie_alg{T<:Number}(A::Matrix{T},
                                  y::VecOrMat{T},
                                  y0::VecOrMat{T};
                                  # Optional
                                  k_max = 100,
                                  B=A',
			                      find_supp = x -> large_ind(vec(sum(abs2,x,2))),
	                              solve_restricted = L -> A[:,L] \ y)
	L = Int[]
    x    = complex(zeros(size(A,2),size(y,2)))
    yhat = zeros(y)
    ytmp = zeros(y)
    rate_proxy = 0.0
    for k=1:k_max
        rate_proxy_prev = rate_proxy

		# Union of support sets
		L = union(L,find_supp(B*(y-A*x)))
		# Restricted LS estimate
        x[L,:] = solve_restricted(L)
        ytmp = A*x

        # Calculate angle between y and ytmp
        rate_proxy = sum([abs2(dot(y0[:,i],ytmp[:,i]))/max(1e-8,sum(abs2,ytmp[:,i])) for i=1:size(y,2)])

        if rate_proxy_prev > rate_proxy
            break;
        end
        yhat[:] = ytmp[:]
    end
	return yhat, L
end
function large_ind(x;k=1::Int)
	return sortperm(x,rev=true)[1:k]
end
