mutable struct ConvNN
    reg_coeff
    learning_rate
    transform
    activation

    y    # observation
    x0   # true state

    biases
    kernels
    x    # estimate

    train_step
    sess

    ConvNN() = new()
end

"""
    ConvNN(nLayers::Integer, nAntennas::Integer; transform = (x,_) -> x, learning_rate=1e-3,reg_coeff=1e-6,activation=nn.relu)

Input and output of the NN are complex-valued vectors of dimension `nCoherence, nFilterLength`
where `nFilterLength` is calculated from `nAntennas` and `transform`.
For mini-batch training, these are stacked into three dimensional arrays of dimension
`nBatches, nCoherence, nFilterLength`.
This method performs a random initialization of all kernels and biases.
"""
function ConvNN(nLayers::Integer, nAntennas::Integer; transform = (x,_) -> x, learning_rate=1e-3,reg_coeff=1e-6,activation=nn.relu)

    nFilterLength = size(transform(eye(nAntennas), :notransp),1)
    
    kernels = [rand(TruncatedNormal(0,0.1,-0.3,0.3), nFilterLength) for i in 1:nLayers]
    biases  = [0.1*ones(nFilterLength) for i in 1:nLayers]
    
    return ConvNN(kernels, biases, transform = transform, learning_rate=learning_rate, reg_coeff=reg_coeff, activation=activation)
end

function ConvNN(kernels::Array, biases::Array; transform = (x,_) -> x, learning_rate=1e-3,reg_coeff=1e-6,activation=nn.relu)
    nLayers = length(kernels)
    kernels_tf = Array{TensorFlow.Variables.Variable}(nLayers)
    biases_tf  = Array{TensorFlow.Variables.Variable}(nLayers)
    for i in 1:nLayers
        kernels_tf[i] = Variable(Float32.(kernels[i]))
        biases_tf[i]  = Variable(Float32.(biases[i]))
    end
    return ConvNN(kernels_tf, biases_tf; transform = transform, learning_rate=learning_rate,reg_coeff=reg_coeff,activation=activation)

end
# This method takes set values as input for kernels and biases
function ConvNN(kernels::Array{TensorFlow.Variables.Variable}, biases::Array{TensorFlow.Variables.Variable}; transform = (x,_) -> x, learning_rate=1e-3,reg_coeff=1e-6,activation=nn.relu)
    est = ConvNN()
    const nBatches = -1 # special value in TensorFlow (nBatches not known at this point and may also vary during training etc.)

    est.reg_coeff     = reg_coeff
    est.learning_rate = learning_rate
    est.transform     = transform
    est.activation    = activation    
    
    est.y             = placeholder(Complex64) # dims: nBatches, nCoherence, nFilterLength
    est.x0            = placeholder(Complex64) # dims: nBatches, nCoherence, nFilterLength

    est.kernels = kernels
    est.biases  = biases
    nLayers = length(kernels)
    nFilterLength = get(get_shape(kernels[1]).dims[1])

    # Build NN
    intermediates = Array{TensorFlow.Tensor}(nLayers)

    # For each batch b=1,...,nBatches, the input to the network is given as sum_t |y[b,t,:]|^2
    intermediates[1] = reduce_mean(square(real(est.y)) + square(imag(est.y)), axis=[2]) # dims: nBatches, nFilterLength
    for i in 1:nLayers-1
        intermediates[i+1] = activation( circ_conv(intermediates[i], kernels[i]) + biases[i] ) # dims: nBatches, nFilterLength
    end
    # Last layer is without activation function
    filt = circ_conv(intermediates[end], kernels[end]) + biases[end] # dims: nBatches, nFilterLength

    # The output of the NN is an element-wise multiplication of the filter with the input data
    est.x = reshape(filt, [nBatches, 1, nFilterLength]) .* est.y # dims: nBatches, nCoherence, nFilterLength

    # Set learning parameters
    regu = reduce_mean(square(kernels[1])) # regularization term
    for i in 2:nLayers
        regu = regu + reduce_mean(square(kernels[i]))
    end
    cost_function  = reduce_mean(square(real(est.x-est.x0)) + square(imag(est.x-est.x0))) + (est.reg_coeff*regu)
    opt_alg        = tf.train.AdamOptimizer(est.learning_rate)
    est.train_step = tf.train.minimize(opt_alg,cost_function)

    est.sess = Session()
    init = global_variables_initializer()
    run(est.sess,init)

    # close TF-session when network is de-referenced
    # note: All nodes remain in the graph. To reset the graph, restart julia.
    finalizer(est, est -> close(est.sess))
    est
end

# This method takes a (smaller) network as input and interpolates
# the values of the kernels and biases to the (larger) new filter length.
# The TF session of the smaller network is then closed.
function resize( est::ConvNN, nAntennas::Integer; learning_rate=1e-3,reg_coeff=1e-6 )

    nLayers          = length(est.kernels)
    nFilterLength    = size(est.transform(eye(nAntennas), :notransp),1)
    nFilterLengthOld = get(get_shape(est.kernels[1]).dims[1])    
    kernels = Array{TensorFlow.Variables.Variable}(nLayers)
    biases  = Array{TensorFlow.Variables.Variable}(nLayers)
    for i in 1:nLayers
        kernel_old       = run(est.sess, est.kernels[i])
        bias_old         = run(est.sess, est.biases[i])

        kernel_old_itp = interpolate(kernel_old, BSpline(Quadratic(Line())), OnGrid())
        bias_old_itp   = interpolate(bias_old,   BSpline(Quadratic(Line())), OnGrid())

        # normalize kernel such that its energy remains the same
        kernel = kernel_old_itp[linspace(1,nFilterLengthOld,nFilterLength)] .* nFilterLengthOld/nFilterLength
        bias   = bias_old_itp[  linspace(1,nFilterLengthOld,nFilterLength)]
        
        kernels[i] = Variable(Float32.(kernel[:])) # dims: nFilterLength
        biases[i]  = Variable(Float32.(bias[:]))   # dims: nFilterLength
    end

    # create new network and overwrite reference to old network
    ConvNN(kernels, biases, transform = est.transform, learning_rate=learning_rate, reg_coeff=reg_coeff, activation=est.activation)
end

function train!(est::ConvNN, y, x0)
    run(est.sess, est.train_step, Dict([(est.y, cnn_perm(est.transform(y,:notransp))), (est.x0, cnn_perm(est.transform(x0,:notransp)))]))
end
function estimate(est::ConvNN, y)
    x = est.transform( cnn_perm(run(est.sess, est.x, Dict([(est.y, cnn_perm(est.transform(y,:notransp)))]))), :transp)
end

#
# The TensorFlow.Saver functions are somewhat difficult to use with Julia.
# Julia can handle only a single graph, hence all networks are defined in
# the same graph. Thus, loading and saving always concerns all networks
# (if multiple networks are defined)
#
# function save(est::ConvNN, filename)
#     saver = tf.train.Saver()
#     tf.train.save(saver, est.sess, filename)
# end
# function load(est::ConvNN, filename)
#     if isfile(filename)
#         saver = tf.train.Saver()
#         tf.train.restore(saver, est.sess, filename)
#     else
#         warn(@sprintf "Ignoring load... File %s not found" filename)
#     end
# end

function save(est::ConvNN, filename)
    nLayers = length(est.kernels)
    nFilterLength = get(get_shape(est.kernels[1]).dims[1])
    kernels = zeros(nFilterLength,nLayers)
    biases  = zeros(nFilterLength,nLayers)
    for i in 1:nLayers
        kernels[:,i] = run(est.sess, est.kernels[i])
        biases[:,i]  = run(est.sess, est.biases[i])
    end
    data = Dict("kernels" => kernels, "biases" => biases)

    JLD.save(filename, "kernels", kernels, "biases", biases)
end

function load!(est::ConvNN, filename)
    if isfile(filename)
        @printf "Loading %s\n" filename
        kernels_vals = load(filename, "kernels")
        biases_vals  = load(filename, "biases")

        nLayers = length(est.kernels)
        for i in 1:nLayers
            ass = assign(est.kernels[i], Float32.(kernels_vals[:,i]))
            run(est.sess, ass)
            ass = assign(est.biases[i], Float32.(biases_vals[:,i]))
            run(est.sess, ass)        
        end
    else
        warn(@sprintf "Ignoring load... File %s not found" filename)
    end
end
