# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using ForwardDiffPullbacks
using Test

using Distributions
using ForwardDiff, ChainRulesCore, Zygote

using ForwardDiffPullbacks: make_tangent


include("testfuncs.jl")

@testset "Zygote" begin
    function cr_fwd_and_back(f, xs, ΔΩ)
        y, back = ChainRulesCore.rrule(f, xs...)
        back_thunks = back(ΔΩ)
        Δx = map(unthunk, back_thunks)
        @assert first(Δx) == NoTangent()
        y, Base.tail(Δx)
    end

    function zg_fwd_and_back(f, xs, ΔΩ)
        y, back = Zygote.pullback(f, xs...)
        Δx = back(ΔΩ)
        y, Δx
    end

 
    function cr_bc_fwd_and_back(f, Xs, ΔΩA)
        y, back = ChainRulesCore.rrule(Base.broadcasted, f, Xs...)
        back_thunks = back(ΔΩA)
        Δx = map(unthunk, back_thunks)
        @assert first(Δx) == NoTangent()
        y, Base.tail(Δx)
    end

    function zg_bc_fwd_and_back(f, Xs, ΔΩA)
        y, back = Zygote.pullback((Xs...) -> f.(Xs...), Xs...)
        Δx = back(ΔΩA)
        y, Δx
    end
    function zg_bcpre_fwd_and_back(f, Xs, ΔΩA)
        y, back = Zygote.pullback(Base.broadcasted, f, Xs...)
        Δx = back(ΔΩA)
        y, Δx
    end


    @test @inferred(cr_fwd_and_back(fwddiff(f), xs, ΔΩ)) isa Tuple{Tuple{Float32, Float32, Int}, Tuple{Float32, Tangent{Tuple{Int, Float32}, Tuple{Float32, Float32}}, SVector{3, Float32}}}
    @test @inferred(zg_fwd_and_back(fwddiff(f), xs, ΔΩ)) isa Tuple{Tuple{Float32, Float32, Int}, Tuple{Float32, Tuple{Float32, Float32}, SVector{3, Float32}}}

    @test cr_fwd_and_back(fwddiff(f), xs, ΔΩ) == ((139, 783, 42), (280, Tangent{Tuple{Int, Float32}}(600.0f0, 1040.0f0), SVector(1600, 2280, 3080)))
    @test zg_fwd_and_back(fwddiff(f), xs, ΔΩ) == ((139, 783, 42), (280, (600, 1040), SVector(1600, 2280, 3080))) # == zg_fwd_and_back(f, xs, ΔΩ)


    # @inferred cr_bc_fwd_and_back(fwddiff(f), Xs, ΔΩA)
    # zg_bc_fwd_and_back(fwddiff(f), Xs, ΔΩA)
    # zg_bc_fwd_and_back(f, Xs, ΔΩA)


    @testset "disttrafo" begin
        function disttrafo(trg_dist::Distribution, src_dist::Distribution, x::Real, prev_ladj::Union{Real,Missing})
            # @info "disttrafo for real x and optional prev_ladj"
            u = cdf(src_dist, x)
            y = quantile(trg_dist, u)
            ladj = if isnothing(prev_ladj)
                nothing
            else
                logladj_1 = + logpdf(src_dist, x)
                logladj_2 = - logpdf(trg_dist, y)
                logladj_1 + logladj_2 + prev_ladj
            end
            (y = y, ladj = ladj)
        end
    
        function disttrafo(trg_dist::Distribution, src_dist::Distribution, x_dual::ForwardDiff.Dual{TAG}, prev_ladj::Missing) where TAG
            # @info "disttrafo for dual x and missing prev_ladj"
            x = ForwardDiff.value(x_dual)
            u = cdf(src_dist, x)
            y = quantile(trg_dist, u)
            dudx = pdf(src_dist, x)
            dydu = inv(pdf(trg_dist, y))
            y_dual = ForwardDiff.Dual{TAG}(x, dydu * dudx * ForwardDiff.partials(x_dual))
            (y = y_dual, ladj = nothing)
        end
    
        # Must work:
        ForwardDiff.derivative(x -> disttrafo(Beta(), Normal(), x, missing).y, -0.5)
    
        # Requires Zygote to utilize thunks (https://github.com/FluxML/Zygote.jl/pull/966):
        if isdefined(Zygote, Symbol("@_adjoint_keepthunks"))
            Zygote.gradient(x -> fwddiff(disttrafo)(Beta(), Normal(), x, missing).y, -0.5)
        end
    
        # Expected to fail for Beta():
        # Zygote.gradient(x -> disttrafo(Beta(), Normal(), x, missing).y, x)
    
    
        # disttrafo(trg_dist::Distribution, src_dist::Distribution, x::Dual, prev_ladj::Missing) = (y = quantile(trg_dist, cdf(src_dist, x)), ladj = missing)
    
        trg_d = Weibull()
        src_d = Normal()
        x = 0.5
    
        ΔΩ = (y = 1, ladj = nothing)
        #=@inferred=# cr_fwd_and_back(fwddiff(disttrafo), (trg_d, src_d, x, missing), ΔΩ)
        #=@inferred=# zg_fwd_and_back(fwddiff(disttrafo), (trg_d, src_d, x, missing), ΔΩ)
        #compare with (ignore missings):
        zg_fwd_and_back(disttrafo, (trg_d, src_d, x, missing), ΔΩ)
    
        ΔΩ = (y = 1, ladj = 42)
        #=@inferred=# cr_fwd_and_back(fwddiff(disttrafo), (trg_d, src_d, x, 7), ΔΩ)
        #=@inferred=# zg_fwd_and_back(fwddiff(disttrafo), (trg_d, src_d, x, 7), ΔΩ)
        #compare with (use deep approx):
        zg_fwd_and_back(disttrafo, (trg_d, src_d, x, 7), ΔΩ)
    
    
        n = 100
        trg_D = fill(trg_d, n)
        src_D = fill(src_d, n)
        X = randn(n)
    
        ΔΩs = fill((y = 1, ladj = nothing), n)
        #=@inferred=#(cr_bc_fwd_and_back(fwddiff(disttrafo), (trg_D, src_D, X, missing), ΔΩs))
        #=@inferred=#(zg_bc_fwd_and_back(fwddiff(disttrafo), (trg_D, src_D, X, missing), ΔΩs))
        #compare with (ignore missings):
        zg_bc_fwd_and_back(disttrafo, (trg_D, src_D, X, missing), ΔΩs)
    
        ΔΩs = fill((y = 1, ladj = 42), n)
        #=@inferred=#(cr_bc_fwd_and_back(fwddiff(disttrafo), (trg_D, src_D, X, 7), ΔΩs))
        #=@inferred=#(zg_bc_fwd_and_back(fwddiff(disttrafo), (trg_D, src_D, X, 7), ΔΩs))
        #compare with (use deep approx):
        zg_bc_fwd_and_back(disttrafo, (trg_D, src_D, X, 7), ΔΩs)
    
        #=
        using BenchmarkTools
        @benchmark disttrafo.($trg_D, $src_D, $X, $7)
        @benchmark cr_bc_fwd_and_back(fwddiff(disttrafo), ($trg_D, $src_D, $X, 7), $ΔΩs)
        @benchmark zg_bc_fwd_and_back(fwddiff(disttrafo), ($trg_D, $src_D, $X, 7), $ΔΩs)
        @benchmark zg_bc_fwd_and_back(disttrafo, ($trg_D, $src_D, $X, 7), $ΔΩs)
        =#
    
    
        function dummy_loss(trg_d::Distribution, src_d::Distribution, x::Real, prev_ladj::Union{Real,Missing})
            y, ladj = fwddiff(disttrafo)(trg_d, src_d, x, prev_ladj)
            ismissing(ladj) ? 2 * y : typeof(y)(y^2 + ladj^2)
        end
    
        #=@inferred=# Zygote.gradient(dummy_loss, trg_d, src_d, x, 42)
        #=@inferred=# Zygote.gradient(dummy_loss, trg_d, src_d, x, missing)
    
        function bc_dummy_loss(trg_D::AbstractVector{<:Distribution}, src_D::AbstractVector{<:Distribution}, X::AbstractVector{<:Real}, prev_ladj::Union{Real,AbstractVector{<:Real},Missing})
            Y_ladj = fwddiff(disttrafo).(trg_D, src_D, X, prev_ladj)
            Y = (x -> x.y).(Y_ladj)
            ladj = (x -> x.ladj).(Y_ladj)
            T = eltype(Y)
            any(ismissing, ladj) ? T(sum(Y)) : T(sum(Y)^2 + sum(ladj)^2)
        end
    
        #=@inferred=# bc_dummy_loss(trg_D, src_D, X, 42)
        #=@inferred=# bc_dummy_loss(trg_D, src_D, X, missing)
    
        #=@inferred=# Zygote.gradient(bc_dummy_loss, trg_D, src_D, X, 42)
        #=@inferred=# Zygote.gradient(bc_dummy_loss, trg_D, src_D, X, missing)
    
        #=@inferred=# Zygote.gradient(X -> bc_dummy_loss(trg_D, src_D, X, 42), X)
    
        # Requires Zygote to utilize thunks (https://github.com/FluxML/Zygote.jl/pull/966):
        if isdefined(Zygote, Symbol("@_adjoint_keepthunks"))
            trg_D = fill(Beta(), n)
            Zygote.gradient(X -> bc_dummy_loss(trg_D, src_D, X, missing), X)
        end
    end

    @testset "struct ctor pullbacks" begin
        mkdists(Mean, Sigma) = Normal.(Mean, Sigma)
        mkdists_fd(Mean, Sigma) = fwddiff(Normal).(Mean, Sigma)

        getlogpdfs(Mean, Sigma, X) = logpdf.(mkdists(Mean, Sigma), X)
        getlogpdfs_fd(Mean, Sigma, X) = fwddiff(logpdf).(mkdists_fd(Mean, Sigma), X)

        sum_getlogpdfs(Mean, Sigma, X) = sum(getlogpdfs(Mean, Sigma, X))
        sum_getlogpdfs_fd(Mean, Sigma, X) = sum(getlogpdfs_fd(Mean, Sigma, X))

        Sigma = rand(10)
        Mean = randn(10)
        X = rand.(mkdists(Mean, Sigma))

        @test @inferred(getlogpdfs_fd(Mean, Sigma, X)) == getlogpdfs(Mean, Sigma, X)

        @test all(isapprox.(Zygote.gradient(sum_getlogpdfs, Mean, Sigma, X), Zygote.gradient(sum_getlogpdfs_fd, Mean, Sigma, X)))
    end
end
