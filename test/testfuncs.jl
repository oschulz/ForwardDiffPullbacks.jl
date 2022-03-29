# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

import Test

import ForwardDiffPullbacks

using LinearAlgebra, StaticArrays

sum_pow2(x) = sum(map(x -> x^2, x))
sum_pow3(x) = sum(map(x -> x^3, x))
f = (xs...) -> (sum(map(sum_pow2, xs)), sum(map(sum_pow3, xs)), 42)
xs = (2f0, (3, 4f0), SVector(5f0, 6f0, 7f0))
ΔΩ = (10, 20, 30)

fy = let f = f; (xs...) -> SVector(f(xs...)); end
ΔΩy = SVector(ΔΩ)


function f_loss_1(xs...)
    r = fwddiff(f)(xs...)
    @assert sum(r) < 10000
    sum(r[1])
end

function f_loss_3(xs...)
    r = fwddiff(f)(xs...)
    @assert sum(r) < 10000
    sum(r[3])
end

function f_loss_3z(xs...)
    r = f(xs...)
    @assert sum(r) < 10000
    sum(r[3])
end


const_foo(x, xs...) = 42.0

function scalar_foo(x, xs...)
    x_flat = ForwardDiffPullbacks.flatten(x)
    dot(x_flat, reverse(x_flat)) * dot(x_flat, x_flat)
end

function nt_foo(x, xs...)
    x_flat = ForwardDiffPullbacks.flatten(x)
    a = dot(x_flat, reverse(x_flat)) * dot(x_flat, x_flat)
    c = SVector(dot(x_flat, reverse(x_flat)), 4.2)
    d = (a^2, a^3)
    e = 42.0
    (a = a, b = (c = d, d = d), e = e)
end

scalar_x = 1.1
tpl_x = (1.1, 2.2)
svec_x = SVector(1.1, 2.2, 3.3)
nt_x = (k = 1.1, l = (c = SVector(2.2, 3.3), m = (4.4, 5.5)))
