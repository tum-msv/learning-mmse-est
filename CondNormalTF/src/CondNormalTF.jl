"""
This module provides some neural-network-based estimators of the following form:

Input: Vectors y[1,:], ..., y[nTrain,:]

A summary statistic z is calculated from the inputs and this is the input to the NN.
The output of the NN is an "element-wise" filter w, which is then applied to each input y[t,:]

Input filter estimator: vector z[:] = sum_t |y[t,:]|.^2
Output filter: vector w[:]

Output: Vectors y[1,:] .* w[:], ..., y[nTrain,:] .* w[:]

Note: For uplink channel estimation, the input data y[t,:] (of length #antennas)
should be the DFT of what is observed at the antennas. The output of the NN are then
the DFTs of the estimated spatial channels.
"""
module CondNormalTF

using TensorFlow; const tf = TensorFlow
using Distributions
using Interpolations

square = import_op("Square")
"""
    circ_conv(x,w)

Calculate circular convolution of each vector `x[b,:]` with filter `w[:]`
of same length for each `b=1,...,nBatches`.

Note that nn.conv2d calculates the linear convolution with the time-reversed
filter (=correlation). Thus, the filter w needs to be reversed and repeated.
"""
function circ_conv(x,w)
    # x,y dims: nBatches, nFilterLength
    const nBatches = -1 # special value in TF
    s     = get_shape(w).dims
    nFilterLength = get(s[1])

    xx = reshape(x,[nBatches,nFilterLength,1,1]) # arrange as 4D array for conv2d
    ww = reshape(tile(w[nFilterLength:-1:1],[2]), [2*nFilterLength,1,1,1]) # reverse, repeat, and arrange as 4D-array for conv2d
    
    z = squeeze(nn.conv2d(xx,ww, strides=[1,1,1,1], padding="SAME"),[3,4]) # dims: nBatches, nFilterLength
end

# TF routines need format: batch-coherence-antennas
# and CNN operates on DFT of observations
cnn_perm = x -> permutedims(x,[3,2,1])


include("ConvNN.jl")


end
