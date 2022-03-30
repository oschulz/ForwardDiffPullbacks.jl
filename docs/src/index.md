# ForwardDiffPullbacks.jl

ForwardDiffPullbacks implements pullbacks compatible with [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) that are calculated via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl).

This package provides the function [`fwddiff`](@ref). If wrapped around a function (i.e. `fwddiff(f)`), it will cause ChainRules (and implicitly Zygote) pullbacks to be calculated using ForwardDiff (i.e. by evaluating the original function with `ForwardDiff.Dual` numbers, possibly multiple times). The pullback will return a ChainRule thunk for each argument of the function.

So `Zygote.gradient(fwddiff(f), xs...)` should yield the same result as `Zygote.gradient(f, xs...)`, but will typically be substantially faster for a function that has a comparatively small number of arguments, especially if the function runs a deep calculation.

ForwardDiffPullbacks does come with broadcasting support, `fwddiff(f).(args...)` will use ForwardDiff to differentiate each iteration in the broadcast separately.

Currently, ForwardDiffPullbacks supports functions whose arguments and result(s) are statically sized, like `Real`, `Tuple`, `StaticArrays.StaticArray` and (nested) `NamedTuple`s and plain structs. Dynamic arrays are not supported yet.
