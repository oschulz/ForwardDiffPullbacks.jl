# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using LinearAlgebra, StaticArrays

sum_pow2(x) = sum(map(x -> x^2, x))
sum_pow3(x) = sum(map(x -> x^3, x))
f = (xs...) -> (sum(map(sum_pow2, xs)), sum(map(sum_pow3, xs)), 42)
xs = (2f0, (3, 4f0), SVector(5f0, 6f0, 7f0))
ΔΩ = (10, 20, 30)

fy = let f = f; (xs...) -> SVector(f(xs...)); end
ΔΩy = SVector(ΔΩ)
