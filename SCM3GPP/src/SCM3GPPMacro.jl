type SCM3GPPMacro
    # Angle spread
    mu_AS::Real
    eps_AS::Real
    r_AS::Real

    pathAS::Real

    # Delay spread
    mu_DS::Real
    eps_DS::Real
    r_DS::Real

    eps_PL::Real # 10*path loss exponent

    nPaths::Integer

    SCM3GPPMacro() = new()
end

function urbanMacro15Deg()
    scen = SCM3GPPMacro()

    # see Table 5.1 in 3GPP TR v12, p.17
    scen.mu_AS = 1.18 
    scen.eps_AS = 0.21
    scen.r_AS = 1.3   

    scen.pathAS = 2.0  

    scen.mu_DS = -6.18
    scen.eps_DS = 0.18
    scen.r_DS = 1.7   

    scen.eps_PL = 35  
    scen.nPaths = 6   

    return scen
end
    
function generate_channel(scen::SCM3GPPMacro, nAntennas; nCoherence=1,nBatches=1)
    h = zeros(Complex128,nAntennas, nCoherence, nBatches)
    t = zeros(Complex128,nAntennas, nBatches)
    for i in 1:nBatches

        # user angle
        theta = (rand()-0.5)*120

        # path delays
        DS = 10.^(scen.mu_DS + scen.eps_DS*randn());
        Tc = 1/3.84e6;
        tau = -scen.r_DS*DS*log.(rand(scen.nPaths));
        tau = sort(tau) - minimum(tau);
        tau_quant = Tc/16*floor.(tau/Tc*16 + 0.5);

        # path powers
        exponent = -1./DS*(scen.r_DS-1)/scen.r_DS;
        Z = randn(scen.nPaths)*3; #per path shadow fading in dB
        p = exp.(exponent*tau).*(10.^(0.1*Z));
        p = p./sum(p);

        # path AoDs
        AS = 10.^(scen.mu_AS + scen.eps_AS*randn());
        aodsm = randn(scen.nPaths)*scen.r_AS*AS;
        ixs = sortperm(abs.(aodsm));
        aodsm = aodsm[ixs];


        (h[:,:,i],t[:,i]) = scm_channel(theta .+ aodsm, p, nAntennas; nCoherence=nCoherence,AS=scen.pathAS)

        # pathloss
        minDist = 1000
        maxDist = 1500
        distance = rand() * (maxDist-minDist) + minDist 
        PL = scen.eps_PL*log10(distance/maxDist)
        beta = 10^(-0.1*PL)

        h[:,:,i] .*= sqrt(beta)
        t[:,i] .*= beta
    end
    return (h,t)
end
