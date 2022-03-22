# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

import ForwardDiffPullbacks
import Test

import ChainRulesCore, ChainRulesTestUtils
import ForwardDiff


nested_cmp(a::ChainRulesCore.AbstractZero, b::ChainRulesCore.AbstractZero) = true
nested_cmp(a::ChainRulesCore.AbstractZero, b) = all(iszero, ForwardDiffPullbacks.flatten(b))
nested_cmp(a, b::ChainRulesCore.AbstractZero) = nested_cmp(b, a)
nested_cmp(a::Real, b::Real) = a ≈ b
nested_cmp(a::AbstractVector, b::AbstractVector) = all(map(nested_cmp, a, b))
nested_cmp(a::Tuple, b::Tuple) = all(map(nested_cmp, a, b))
nested_cmp(a::NamedTuple{names}, b::NamedTuple{names}) where names = nested_cmp(values(a), values(b))
nested_cmp(a::ChainRulesCore.Tangent{T}, b::ChainRulesCore.Tangent{T}) where T = nested_cmp(ChainRulesCore.backing(a), ChainRulesCore.backing(b))


# For cases that test_rrule can't handle (yet):
function custom_test_rrule(f::Function, xs...)
    Test.@testset "custom_test_rrule $(typeof(f)) for $(typeof(xs))" begin
        xs_flat = ForwardDiffPullbacks.flatten(xs)

        f_flatin = xs_flat -> f(ForwardDiffPullbacks.unflatten(xs, xs_flat)...)
        f_flatio = ForwardDiffPullbacks.flatten ∘ f_flatin

        J = ForwardDiff.jacobian(f_flatio, xs_flat)
        y = f(xs...)
        y_flat = f_flatio(xs_flat)

        rand_dxs = map(ChainRulesTestUtils.rand_tangent, xs)
        dy = ChainRulesTestUtils.rand_tangent(y)
        dy_flat = ForwardDiffPullbacks.flatten(dy)

        dxs_flat = J' * dy_flat
        dxs = ChainRulesCore.backing(ForwardDiffPullbacks.unflatten(rand_dxs, dxs_flat))
        dfxs = (NoTangent(), dxs...)
    
        Test.@test typeof(f_flatin(xs_flat)) == typeof(f(xs...))
        Test.@test ForwardDiffPullbacks.flatten(y) ≈ y_flat

        Test.@test Test.@inferred(ChainRulesCore.rrule(fwddiff(f), xs...)) isa Tuple{typeof(y), Function}
        y_rrule, back = ChainRulesCore.rrule(fwddiff(f), xs...)
        Test.@test y_rrule == y
        Test.@test nested_cmp(dfxs, Test.@inferred(map(unthunk, back(dy))))
    end
end
