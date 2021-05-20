# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    ForwardDiffPullbacks

ChainRulesCore compatible pullbacks using ForwardDiff.
"""
module ForwardDiffPullbacks

using LinearAlgebra

import ChainRulesCore
import ForwardDiff
import StaticArrays

# using Requires

include("chain_rules_aliases.jl")
include("dual_numbers.jl")
include("chainrules_types.jl")
include("fwd_back.jl")
include("broadcasting.jl")
include("with_forwarddiff.jl")
include("rrules.jl")

function __init__()
    # Possible extensions:
    # @require Nabla = "49c96f43-aa6d-5a04-a506-44c7070ebe78" include("nabla_support.jl")
    # @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include("reversediff_support.jl")
    # @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("tracker_support.jl")
end

end # module
