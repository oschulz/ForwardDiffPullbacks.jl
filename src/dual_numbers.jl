# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

function dual_number end
function dual_value end
function dual_partials end
function dual_tagtype end

@inline dual_number(::Type{TagType}, x::Real, p::NTuple{N,Real}) where {TagType,N} = ForwardDiff.Dual{TagType}( x, p...)
@inline dual_value(x::Real) = ForwardDiff.value(x)

@inline dual_partials(x::ForwardDiff.Dual) = ForwardDiff.partials(x)
@inline dual_partials(x::Real) = ZeroTangent()

@inline dual_tagtype(f::Any, ::Type{T}) where T = typeof(ForwardDiff.Tag(f, T))
