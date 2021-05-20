# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using ForwardDiffPullbacks
using Test

using ChainRulesCore

include("testfuncs.jl")

@testset "ChainRulesCore" begin
    @test @tinferred(ChainRulesCore.rrule(fwddiff(f), xs...)) isa Tuple{Tuple, Function}
    @test @tinferred((ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ)) isa Tuple{ZeroTangent, ForwardDiffPullbacks.FwdDiffPullbackThunk, ForwardDiffPullbacks.FwdDiffPullbackThunk,ForwardDiffPullbacks.FwdDiffPullbackThunk}
    @test @tinferred(map(unthunk, (ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ))) == (ZeroTangent(), 280, (600, 1040), SVector(1600, 2280, 3080))
end
