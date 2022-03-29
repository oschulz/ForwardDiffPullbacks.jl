# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using ForwardDiffPullbacks
using Test

using ChainRulesCore: Tangent
using ForwardDiffPullbacks: make_tangent

include("testfuncs.jl")

@testset "fwd_back" begin
    sum_pow2(x) = sum(map(x -> x^2, x))
    sum_pow3(x) = sum(map(x -> x^3, x))
    f = (xs...) -> (sum(map(sum_pow2, xs)), sum(map(sum_pow3, xs)), 42)
    xs = (2f0, (3, 4f0), SVector(5f0, 6f0, 7f0))
    ΔΩ = (10, 20, 30)

    fy = let f = f; (xs...) -> SVector(f(xs...)); end
    ΔΩy = SVector(ΔΩ)


    @test @inferred(ForwardDiffPullbacks.forwarddiff_vjp_impl(f, xs, Val(1), ΔΩ)) == 280
    @test @inferred(ForwardDiffPullbacks.forwarddiff_vjp_impl(f, xs, Val(2), ΔΩ)) == Tangent{typeof(xs[2])}(600f0, 1040f0)
    @test @inferred(ForwardDiffPullbacks.forwarddiff_vjp_impl(f, xs, Val(3), ΔΩ)) == SVector(1600, 2280, 3080)

    Xs = map(x -> fill(x, 5), xs)
    ΔΩA = fill(ΔΩ, 5)

    let args = Xs
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(1), Ref(ΔΩ))) == fill(280f0, 5)
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(2), Ref(ΔΩ))) == fill(Tangent{typeof(xs[2])}(600f0, 1040f0), 5)
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(3), Ref(ΔΩ))) == fill(SVector(1600, 2280, 3080), 5)
    end

    let args = (Xs[1][1], Ref(xs[2]), Xs[3])
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(1), Ref(ΔΩ))) == 5 * 280f0
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(2), Ref(ΔΩ))) == make_tangent(typeof(args[2]), (x = make_tangent(typeof(args[2][]), (5*600f0, 5*1040f0)),))
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(3), Ref(ΔΩ))) == fill(SVector(1600, 2280, 3080), 5)
    end

    let args = map(Ref, xs), ΔY = Ref(ΔΩ)
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(1), Ref(ΔΩ))) == make_tangent(typeof(args[1]), (x = 280,))
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(2), Ref(ΔΩ))) == make_tangent(typeof(args[2]), (x = make_tangent(typeof(args[2][]), (600f0, 1040f0)),))
        @test @inferred(ForwardDiffPullbacks.forwarddiff_bc_vjp_impl(f, args, Val(3), Ref(ΔΩ))) == make_tangent(typeof(args[3]), (x = SVector(1600, 2280, 3080),))
    end
end
