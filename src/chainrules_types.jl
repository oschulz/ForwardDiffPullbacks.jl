# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


const ZeroLike = Union{ChainRulesCore.AbstractZero, Nothing}
const RealTangentLike = Union{Real, ZeroLike}
const RealDualLike = Union{Real, Nothing, Missing}

@inline make_tangent(::Type{PrimalType}, Δ::BackingType) where {PrimalType,BackingType} = Tangent{PrimalType,BackingType}(Δ)
