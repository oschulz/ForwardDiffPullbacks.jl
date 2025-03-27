var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"DocTestSetup  = quote\n    using ForwardDiffPullbacks\nend","category":"page"},{"location":"api/#Modules","page":"API","title":"Modules","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:module]","category":"page"},{"location":"api/#Types-and-constants","page":"API","title":"Types and constants","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:type, :constant]","category":"page"},{"location":"api/#Functions-and-macros","page":"API","title":"Functions and macros","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Order = [:macro, :function]","category":"page"},{"location":"api/#Documentation","page":"API","title":"Documentation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [ForwardDiffPullbacks]\nOrder = [:module, :type, :constant, :macro, :function]","category":"page"},{"location":"api/#ForwardDiffPullbacks.ForwardDiffPullbacks","page":"API","title":"ForwardDiffPullbacks.ForwardDiffPullbacks","text":"ForwardDiffPullbacks\n\nChainRulesCore compatible pullbacks using ForwardDiff.\n\n\n\n\n\n","category":"module"},{"location":"api/#ForwardDiffPullbacks.fwddiff-Tuple{Union{Function, Type}}","page":"API","title":"ForwardDiffPullbacks.fwddiff","text":"fwddiff(f::Base.Callable)::Function\n\nUse ForwardDiff dual numbers to implement ChainRulesCore pullbacks For\n\n`fwddiff(f)(args...)\n`fwddiff(f).(args...)\n\nExample:\n\nusing ForwardDiffPullbacks, StaticArrays\n\nf = (xs...) -> (sum(map(x -> sum(map(x -> x^2, x)), xs)))\nxs = (2, (3, 4), SVector(5, 6, 7))\nf(xs...) == 139\n\nusing ChainRulesCore\n\ny, back = rrule(fwddiff(f), xs...)\ny == 139\nmap(unthunk, back(1)) == (Zero(), 4, (6, 8), [10, 12, 14])\n\nusing Zygote\n\nZygote.gradient(fwddiff(f), xs...) == Zygote.gradient(f, xs...)\n\nXs = map(x -> fill(x, 100), xs)\nZygote.gradient((Xs...) -> sum(fwddiff(f).(Xs...)), Xs...) ==\n    Zygote.gradient((Xs...) -> sum(f.(Xs...)), Xs...)\n\nThe gradient is the same with and without fwddiff, but fwddiff makes the gradient calculation a lot faster here.\n\n\n\n\n\n","category":"method"},{"location":"LICENSE/#LICENSE","page":"LICENSE","title":"LICENSE","text":"","category":"section"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"using Markdown\nMarkdown.parse_file(joinpath(@__DIR__, \"..\", \"..\", \"LICENSE.md\"))","category":"page"},{"location":"#ForwardDiffPullbacks.jl","page":"Home","title":"ForwardDiffPullbacks.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ForwardDiffPullbacks implements pullbacks compatible with ChainRulesCore that are calculated via ForwardDiff.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package provides the function fwddiff. If wrapped around a function (i.e. fwddiff(f)), it will cause ChainRules (and implicitly Zygote) pullbacks to be calculated using ForwardDiff (i.e. by evaluating the original function with ForwardDiff.Dual numbers, possibly multiple times). The pullback will return a ChainRule thunk for each argument of the function.","category":"page"},{"location":"","page":"Home","title":"Home","text":"So Zygote.gradient(fwddiff(f), xs...) should yield the same result as Zygote.gradient(f, xs...), but will typically be substantially faster a function that has a comparatively small number of arguments, especially if the function runs a deep calculation. Broadcasting (i.e. g.(fwddiff(f))) is supported as well.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Currently, ForwardDiffPullbacks supports functions with Real, Tuple and StaticArrays.SVector arguments. Support for StaticArrays.SArray and Array-valued arguments in general is on the to-do list.","category":"page"}]
}
