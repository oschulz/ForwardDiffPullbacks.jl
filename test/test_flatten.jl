# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using ForwardDiffPullbacks
using Test

using ChainRulesCore
using ForwardDiffPullbacks: flatten, unflatten, unflatten_tangent

include("testfuncs.jl")

@testset "flatten" begin
    @test @inferred(unflatten(scalar_x, flatten(scalar_x))) == scalar_x
    @test @inferred(unflatten(tpl_x, flatten(tpl_x))) == tpl_x
    @test @inferred(unflatten(svec_x, flatten(svec_x))) == svec_x
    @test @inferred(unflatten(nt_x, flatten(nt_x))) == nt_x    

    @test @inferred(unflatten_tangent(scalar_x, flatten(scalar_x))) == scalar_x
    @test @inferred(unflatten_tangent(tpl_x, flatten(tpl_x))) == Tangent{typeof(tpl_x)}(tpl_x...)
    @test @inferred(unflatten_tangent(svec_x, flatten(svec_x))) == svec_x
    @test @inferred(unflatten_tangent(nt_x, flatten(nt_x))) == Tangent{typeof(nt_x)}(;k = nt_x.k, l = Tangent{typeof(nt_x.l)}(c = SVector(2.2, 3.3), m = Tangent{typeof(nt_x.l.m)}(4.4, 5.5)))
end
