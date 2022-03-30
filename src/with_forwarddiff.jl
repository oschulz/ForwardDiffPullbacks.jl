# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


struct AsFunction{F} <: Function
    f::F
end
#AsFunction(f::F) where F = AsFunction{F}(f)  # force specialization
AsFunction(::Type{T}) where T = AsFunction{Type{T}}(T) # For type stability if `f isa UnionAll`

@inline (wrapped_f::AsFunction{F})(xs...) where F = wrapped_f.f(xs...)


@inline asfunction(f) = AsFunction(f)
@inline asfunction(f::Function) = f



struct WithForwardDiff{F<:Function} <: Function
    f::F
end

@inline (wrapped_f::WithForwardDiff{F})(xs...) where F = wrapped_f.f(xs...)

# Desireable for consistent behavior?
# Base.broadcasted(wrapped_f::WithForwardDiff, xs...) = broadcast(wrapped_f.f, xs...)


"""
    fwddiff(f)::Function

Use `ForwardDiff` dual numbers to implement `ChainRulesCore` pullbacks For

* `fwddiff(f)(args...)
* `fwddiff(f).(args...)

Example:

```
using ForwardDiffPullbacks, StaticArrays

f = (xs...) -> (sum(map(x -> sum(map(x -> x^2, x)), xs)))
xs = (2, (3, 4), SVector(5, 6, 7))
f(xs...) == 139

using ChainRulesCore

y, back = rrule(fwddiff(f), xs...)
y == 139
map(unthunk, back(1)) == (NoTangent(), 4, (6, 8), [10, 12, 14])

using Zygote

Zygote.gradient(fwddiff(f), xs...) == Zygote.gradient(f, xs...)

Xs = map(x -> fill(x, 100), xs)
Zygote.gradient((Xs...) -> sum(fwddiff(f).(Xs...)), Xs...) ==
    Zygote.gradient((Xs...) -> sum(f.(Xs...)), Xs...)
```

The gradient is the same with and without `fwddiff`, but `fwddiff` makes the
gradient calculation a lot faster here.
"""
function fwddiff end
export fwddiff

# Wrap functions in AsFunction if necessary for type stability, e.g. if f is a type/ctor:
@inline fwddiff(f) = WithForwardDiff(asfunction(f))
