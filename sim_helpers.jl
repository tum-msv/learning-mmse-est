function crandn(dims...)
    sqrt(0.5) * ( randn(dims) + 1im*randn(dims) )
end
# Define "Q" matrix transformations:
# Q*x  = trans(y)
# Q'*x = trans(y,:transp)

# Q = unitary DFT
function circ_trans(x,tp)
    if tp == :transp
        y = ifft(x,1)*sqrt(size(x,1))
    else
        y = fft(x,1)/sqrt(size(x,1))
    end
    y
end
# Q = first M columns of 2Mx2M unitary DFT
function toep_trans(x,tp)
    if tp == :transp
        y = ifft(x,1)[1:Int(end/2),:,:]*sqrt(size(x,1))
        if length(size(x)) == 2
            y = y[:,:,1]
        end
    else
        y = fft([x;zeros(x)],1)/sqrt(2*size(x,1))
    end
    y
end


function train!(nn_est; snr = 0, nBatches = 1, get_channel = () -> 0.0, verbose = false)
    verbose && @printf "Learning: "
    for b in 1:nBatches
        verbose && mod(b,ceil(Int,nBatches/10))==0 && @printf " ... %.0f%%" b/nBatches*100

        (h,h_cov) = get_channel()
        y         = h + 10^(-snr/20) * crandn(size(h)...)
        for (_,nn) in nn_est
            cntf.train!(nn,y,h)
        end
    end
    verbose && @printf "\n"
end

function init_hier!(nn_est, init_params; verbose = false)
    for i in 1:length(init_params[:nBatches])
        verbose && @printf "Hierarchical Learning %.0f/%.0f\n" i length(init_params[:nBatches])
        nAntennas  = init_params[:nAntennas][i]
        nBatches   = init_params[:nBatches][i]
        nBatchSize = init_params[:nBatchSize][i]
        snr        = init_params[:snrs][i]
        train!(nn_est, snr = snr, nBatches = nBatches, get_channel = () -> init_params[:get_channel](nAntennas, nBatchSize), verbose = verbose)
        for (alg,nn) in nn_est
            nn_est[alg] = cntf.resize(nn, init_params[:nAntennas][i+1], learning_rate = init_params[:learning_rates][alg][i+1])
        end
    end
end

function evaluate(algs; snr = 0, nBatches = 1, get_channel = () -> 0.0, verbose = false)
    errs  = Dict{Symbol,Any}()
    rates = Dict{Symbol,Any}()

    rho = 10^(0.1*snr);

    for alg in keys(algs)
        errs[alg]  = 0.0
        rates[alg] = 0.0
    end

    # Generate channels, calculate errors and achievable rates
    verbose && @printf "Simulating: "
    for bb in 1:nBatches
        verbose && mod(bb,ceil(Int,nBatches/10))==0 && @printf " ... %.0f%%" bb/nBatches*100        
        (h,h_cov) = get_channel()
        y   = h + 10^(-snr/20) * crandn(size(h)...)
        (nAntennas,nCoherence,nBatchSize) = size(h)
        for (alg,est) in algs
            hest = est(y,h,h_cov)
            errs[alg] += sum(abs2,h-hest)/length(h)/nBatches
            # Achievable rates
            for b in 1:nBatchSize, t in 1:nCoherence
                if sum(abs2,hest[:,t,b]) < 1e-8
                    verbose && warn(string(alg) * ": channel estimate is zero")
                    continue
                end
                
                rates[alg] += log2(1 + rho*abs2(dot(h[:,t,b],hest[:,t,b]))/max(1e-8,sum(abs2,hest[:,t,b])))/length(h[1,:,:])/nBatches
            end
        end
    end
    verbose && @printf "\n"
    (errs, rates)
end
