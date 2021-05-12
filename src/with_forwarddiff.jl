# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


struct WithForwardDiff{F} <: Function
    f::F
end

@inline (wrapped_f::WithForwardDiff)(xs...) = wrapped_f.f(xs...)

# Desireable for consistent behavior?
# Base.broadcasted(wrapped_f::WithForwardDiff, xs...) = broadcast(wrapped_f.f, xs...)

"""
    fwddiff(f::Base.Callable)::Function

Use `ForwardDiff` in `ChainRulesCore` pullback For

* `fwddiff(f)(args...)
* `fwddiff(f).(args...)
"""
fwddiff(f::Base.Callable) = WithForwardDiff(f)
export fwddiff
