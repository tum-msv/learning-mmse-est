function crandn(dims...)
    sqrt(0.5) * ( randn(dims) + 1im*randn(dims) )
end
"""
    scm_channel(angles,weights,nAntennas::Integer;nCoherence=1,AS=2.0)

Generate `nCoherence` channel realization according to angular power density (spectrum) `f`
where `f` is given as a superposition of weighted Laplace kernels located at `angles` and
with standard deviation `AS`
The spectrum `f` defines a covariance matrix of the channel `h`.
The first row of this covariance matrix is returned as second output.
"""
function scm_channel(angles,weights,nAntennas::Integer;nCoherence=1,AS=2.0)
    f(x)  = spectrum(x,angles,weights;AS=AS)
    (h,t) = chan_from_spectrum(f,nAntennas,nCoherence=nCoherence)
end


"""
    chan_from_spectrum(f,nAntennas::Integer;nCoherence=1)

Generate `nCoherence` channel realization according to angular power density (spectrum) `f`
(which is a cont. function).
The spectrum `f` defines a covariance matrix of the channel `h`.
The first row of this covariance matrix is returned as second output.
"""
function chan_from_spectrum(f,nAntennas::Integer;nCoherence=1)
    OF = 20 # oversampling factor (ideally, would use continuous freq. spectrum...)
    nFreqSamples = OF*nAntennas
    fs = Vector{Float64}([f(i/nFreqSamples*2*pi) for i=0:nFreqSamples-1])

    # avoid instabilities due to almost infinite energy at some frequencies
    # (this should only happen at "endfire" of a uniform linear array where --
    # because of the arcsin-transform -- the angular psd grows to infinity
    almostInfThreshold = max(1,nFreqSamples) # use nFreqSamples as threshold value...
    almostInfFreqs = abs.(fs) .> almostInfThreshold
    fs[almostInfFreqs] = almostInfThreshold * angle.(fs[almostInfFreqs])

    if sum(fs) > 0
        fs = fs./sum(fs)*nFreqSamples # normalize energy
    end

    x = crandn(nFreqSamples,nCoherence);
    h = ifft(sqrt.(fs).*x,1).*sqrt(nFreqSamples)
    h = h[1:nAntennas,:]

    # t is the first row of the covariance matrix of h (which is Toeplitz and Hermitian)
    t = fft(fs)./nFreqSamples
    t = t[1:nAntennas]
    (h,t)
end


"""
    spectrum(x_rad,angles,weights,AS=2.0)

Generate spectrum at values `asin(x_rad/pi) [deg]` corresponding
to superposition of weighted Laplace kernels with means `angles`
and standard deviations `AS`.
This functions takes care of the folding that happens when transforming
from [-180,180] deg. to [rad]-space...
"""
function spectrum(x_rad,angles,weights;AS=2.0)
    x_rad = mod(x_rad + pi,2*pi)-pi
    x_deg = asin(x_rad/pi)./pi*180

    v =     laplace(    x_deg,angles,weights,AS=AS)
    v = v + laplace(180-x_deg,angles,weights,AS=AS)
    v = v .* (180/pi*2*pi./sqrt(pi^2 - x_rad.^2))
end


"""
    laplace(x_deg::Float64,angles,weights;AS=2.0)

Value at `x_deg` (in deg) of sum of weighted Laplace kernels
(kernels are wrapped to [-180,180] degrees).
"""
function laplace(x_deg::Float64,angles,weights;AS=2.0)
    v = 0.0;
    for i in 1:length(weights)
        xshifted = x_deg - angles[i]
        xshifted = mod(xshifted + 180,360)-180;
        v += weights[i]/(2*AS)*exp(-abs(xshifted)/AS)
    end
    v
end

"""
    laplace(x_deg::Vector{Float64},angles,weights;AS=2.0)

Value at `x_deg` (in deg) of sum of weighted Laplace kernels
(kernels are wrapped to [-180,180] degrees).
"""
function laplace(x_deg::Vector{Float64},angles,weights;AS=2.0)
    v = zeros(x_deg)
    for j in 1:length(x_deg)
        vj = 0.0;
        for i in 1:length(weights)
            xshifted = x_deg[j] - angles[i]
            xshifted = mod(xshifted + 180,360)-180;
            vj += weights[i]/(2*AS)*exp(-abs.(xshifted)/AS)
        end
        v[j] = vj;
    end
    v
end
