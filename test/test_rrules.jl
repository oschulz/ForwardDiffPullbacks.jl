# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using ForwardDiffPullbacks
using Test

using ChainRulesCore

include("testfuncs.jl")

@testset "ChainRulesCore" begin
    @test @inferred(ChainRulesCore.rrule(fwddiff(f), xs...)) isa Tuple{Tuple, Function}
    @test @inferred((ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ)) isa Tuple{ChainRulesCore.Zero, ForwardDiffPullbacks.FwdDiffPullbackThunk, ForwardDiffPullbacks.FwdDiffPullbackThunk,ForwardDiffPullbacks.FwdDiffPullbackThunk}
    @test @inferred(map(unthunk, (ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ))) == (Zero(), 280, (600, 1040), SVector(1600, 2280, 3080))
end
