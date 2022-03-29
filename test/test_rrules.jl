# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using ForwardDiffPullbacks
using Test

using ChainRulesCore, ChainRulesTestUtils
import ForwardDiff

include("testutils.jl")
include("testfuncs.jl")


@testset "ChainRulesCore" begin
    @test @inferred(ChainRulesCore.rrule(fwddiff(f), xs...)) isa Tuple{Tuple, Function}
    @test @inferred((ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ)) isa Tuple{NoTangent, ForwardDiffPullbacks.FwdDiffPullbackThunk, ForwardDiffPullbacks.FwdDiffPullbackThunk,ForwardDiffPullbacks.FwdDiffPullbackThunk}
    @test @inferred(map(unthunk, (ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ))) == (NoTangent(), 280, Tangent{typeof(xs[2])}(600, 1040), SVector(1600, 2280, 3080))

    test_rrule(fwddiff(const_foo), scalar_x)
    custom_test_rrule(fwddiff(const_foo), tpl_x)
    test_rrule(fwddiff(const_foo), svec_x)
    custom_test_rrule(fwddiff(const_foo), nt_x)
    custom_test_rrule(fwddiff(const_foo), nt_x, svec_x)

    test_rrule(fwddiff(scalar_foo), scalar_x)
    custom_test_rrule(fwddiff(scalar_foo), tpl_x)
    test_rrule(fwddiff(scalar_foo), svec_x)
    custom_test_rrule(fwddiff(scalar_foo), nt_x)
    custom_test_rrule(fwddiff(scalar_foo), nt_x, svec_x)

    custom_test_rrule(fwddiff(nt_foo), scalar_x)
    custom_test_rrule(fwddiff(nt_foo), tpl_x)
    custom_test_rrule(fwddiff(nt_foo), svec_x)
    custom_test_rrule(fwddiff(nt_foo), nt_x)
    custom_test_rrule(fwddiff(nt_foo), nt_x, svec_x)
end
