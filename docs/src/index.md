# ForwardDiffPullbacks.jl

ForwardDiffPullbacks implements pullbacks compatible with [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) that are calculated via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl).

This package provides the function [`fwddiff`](@ref). If wrapped around a function (i.e. `fwddiff(f)`), it will cause ChainRules (and implicitly Zygote) pullbacks to be calculated using ForwardDiff (i.e. by evaluating the original function with `ForwardDiff.Dual` numbers, possibly multiple times). The pullback will return a ChainRule thunk for each argument of the function.

So `Zygote.gradient(fwddiff(f), xs...)` should yield the same result as `Zygote.gradient(f, xs...)`, but will typically be substantially faster a function that has a comparatively small number of arguments, especially if the function runs a deep calculation. Broadcasting (i.e. `g.(fwddiff(f))`) is supported as well.

Currently, ForwardDiffPullbacks supports functions with `Real`, `Tuple` and `StaticArrays.SVector` arguments. Support for `StaticArrays.SArray` and `Array`-valued arguments in general is on the to-do list.
