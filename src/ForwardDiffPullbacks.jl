# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    ForwardDiffPullbacks

ChainRulesCore compatible pullbacks using ForwardDiff.
"""
module ForwardDiffPullbacks

using LinearAlgebra

import ChainRulesCore
import ConstructionBase
import ForwardDiff
import Static
import StaticArrays

using ChainRulesCore: AbstractTangent, Tangent, NoTangent, ZeroTangent, AbstractThunk, unthunk
using Static: static, StaticInt
using StaticArrays: SVector

include("chainrules_types.jl")
include("flatten.jl")
include("dual_numbers.jl")
include("fwd_back.jl")
include("with_forwarddiff.jl")
include("rrules.jl")

end # module
