"""
    crandnToep(t)

Generate `x ~ N_C(0,T)` with Toeplitz cov. `T` that has `t` as
its first row.
"""
function crandnToep(t)
    C = toeplitzHe(t)
    y = sqrtm(C)*crandn(length(t))
end

"""
    toeplitz(t::Vector)

Generate an `n` x `n` Toeplitz matrix `C` with generating vector `t` of length `2n-1`.
`C[1,1] = t[n]`
`C[1,n] = t[2n-1]`
`C[n,1] = t[1]`
"""
function toeplitz(t::Vector)
    n = Int(floor(length(t)/2)+1)

    C = [t[j-i+n] for i=1:n,j=1:n]
end

function applyToep(t,x)
    x2 = [x;zeros(x)]
    t2 = [conj(t);0;t[end:-1:2]]
    y = ifft(fft(t2,1).*fft(x2,1))
    y = y[1:length(x)]
end

"""
    toeplitzHe(t::Vector)

Generate an `n` x `n` Hermitian Toeplitz matrix `C` with generating vector `t` of length `n`.
The vector `t` defines the first row of `C` and `t[1]` must be real.
"""
function toeplitzHe(t::Vector)
    t_full = [ conj(t[end:-1:2]); t[1:end] ]
    C = toeplitz(t_full)
end

"""
    best_circulant_approximation(t::Vector)

If `t` is the length `n` generating vector of a Hermitian Toeplitz matrix `T`, then
`c` is the length `n` generating vector of a Hermitian circulant matrix `C`, which
is the best approximation of `T`.
"""
function best_circulant_approximation(t::Vector)
    n = length(t);
    lins = ((n:-2:-n+2)./n);
    t1 = t;
    t2 = cat(1,[0], conj(reverse(t[2:end])));
    c = 0.5*n*real(ifft(lins .*(t1-t2) + t1 + t2));
end


"""
    best_circulant_approximation(t::Matrix)

If each `t[:,m]` is the length `n` generating vector of a Hermitian Toeplitz matrix `T_m`, then
each `c[:,m]` is the length `n` generating vector of a Hermitian circulant matrix `C_m`, which
is the best approximation of `T_m`.
"""
function best_circulant_approximation(t::Matrix)
    n = size(t,1);
    m = size(t,2);
    lins = ((n:-2:-n+2)./n);
    t1 = t;
    t2 = cat(1,zeros(1,m), conj(flipdim(t[2:end,:],1)));
    c = 0.5*n*real(ifft(lins .*(t1-t2) + t1 + t2,1));
end
