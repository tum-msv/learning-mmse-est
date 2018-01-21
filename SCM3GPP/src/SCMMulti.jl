type SCMMulti
    pathAS::Real          # standard deviation of Laplace
    nPaths::Integer      # number of paths

    SCMMulti(; pathAS=2.0, nPaths=3) = new(pathAS,nPaths)
end

function generate_channel(chan::SCMMulti, nAntennas; nCoherence=1,nBatches=1)
    h = zeros(Complex128,nAntennas, nCoherence, nBatches)
    t = zeros(Complex128,nAntennas, nBatches)
    for i in 1:nBatches
        gains = rand(chan.nPaths)
        gains = gains ./ sum(gains,1)
        angles = (rand(chan.nPaths)-0.5)*180 # path center: uniform [-90,90] deg

        (h[:,:,i],t[:,i]) = scm_channel(angles, gains, nAntennas; nCoherence=nCoherence,AS=chan.pathAS)
    end
    return (h,t)
end
