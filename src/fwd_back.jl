# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = dual_number(TagType, x, (true,))

function forwarddiff_dualized(::Type{TagType}, x::NTuple{N,Real}) where {TagType,N}
    ntuple(j -> dual_number(TagType, x[j], ntuple(i -> i == j, Val(N))), Val(N))
end

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::StaticArrays.SVector) where {TagType} = StaticArrays.SVector(forwarddiff_dualized(TagType, (x...,)))


_fieldvals(x) = ntuple(i -> getfield(x, i), Val(fieldcount(typeof(x))))

@generated function _strip_type_parameters(tp::Type{T}) where T
    nm = T.name
    :($(nm.module).$(nm.name))
end

function forwarddiff_dualized(::Type{TagType}, x::T) where {TagType,T}
    tp = _strip_type_parameters(T)
    fieldvals = _fieldvals(x)
    dual_fieldvals = forwarddiff_dualized(TagType, fieldvals)
    tp(dual_fieldvals...)
end


@inline function forwarddiff_fwd(f::Base.Callable, xs::Tuple, ::Val{i}) where i
    TagType = dual_tagtype((f,Val(i)), eltype(xs[i]))
    xs_i_dual = forwarddiff_dualized(TagType, xs[i])
    xs_dual = ntuple(j -> i == j ? xs_i_dual : xs[j], Val(length(xs)))
    f(xs_dual...)
end


@inline forwarddiff_value(y_dual::Real) = dual_value(y_dual)
@inline forwarddiff_value(y_dual::NTuple{N,Real}) where N = map(dual_value, y_dual)
@inline forwarddiff_value(y_dual::StaticArrays.SVector{N,<:Real}) where N = StaticArrays.SVector(map(dual_value, y_dual))


@inline forwarddiff_back_unshaped(ΔΩ::RealTangentLike, y_dual::Real) = (ΔΩ * dual_partials(y_dual)...,)

partials_prod(y_dual::Real, ΔΩ_i::Real) = dual_partials(y_dual) * ΔΩ_i
partials_prod(y_dual::Any, ΔΩ_i::ZeroLike) = ZeroTangent()


function forwarddiff_back_unshaped(ΔΩ::NTuple{N,RealTangentLike}, y_dual::NTuple{N,RealDualLike}) where N
    (sum(map((ΔΩ_i, y_dual_i) -> partials_prod(y_dual_i, ΔΩ_i), ΔΩ, y_dual))...,)
end

function forwarddiff_back_unshaped(ΔΩ::NamedTuple{names,<:NTuple{N,RealTangentLike}}, y_dual::NamedTuple{names,<:NTuple{N,RealDualLike}}) where {names,N}
    forwarddiff_back_unshaped(values(ΔΩ), values(y_dual))
end

@inline function forwarddiff_back_unshaped(ΔΩ::StaticArrays.SVector{N,<:RealTangentLike}, y_dual::StaticArrays.SVector{N,<:RealDualLike}) where N
    forwarddiff_back_unshaped((ΔΩ...,), (y_dual...,))
end

function forwarddiff_back_unshaped(ΔΩ::Tangent, y_dual::Any)
    stripped_ΔΩ = getfield(ChainRulesCore.canonicalize(ΔΩ), :backing)
    forwarddiff_back_unshaped(stripped_ΔΩ, y_dual)
end


@inline shape_forwarddiff_gradient(::Type{<:Real}, Δx::Tuple{}) = nothing
@inline shape_forwarddiff_gradient(::Type{<:Real}, Δx::Tuple{Real}) = Δx[1]
@inline shape_forwarddiff_gradient(::Type{<:Tuple}, Δx::NTuple{N,Real}) where N = Δx
@inline shape_forwarddiff_gradient(::Type{<:StaticArrays.SVector}, Δx::Tuple{}) = nothing
@inline shape_forwarddiff_gradient(::Type{<:StaticArrays.SVector}, Δx::NTuple{N,Real}) where N = StaticArrays.SVector(Δx)

@inline shape_forwarddiff_gradient(::Type{T}, Δx::Tuple{}) where T = nothing
@inline @generated function shape_forwarddiff_gradient(::Type{T}, Δx::Tuple) where T
    :(NamedTuple{$(fieldnames(T))}(Δx))
end

g_tangent = nothing
g_y_dual = nothing
# For `x::StaticArrays.SVector`, `ForwardDiffPullbacks.forwarddiff_back(StaticArrays.SVector, ΔΩ, ForwardDiffPullbacks.forwarddiff_fwd(f, (x,), Val(1))) == ForwardDiff.jacobian(f, x)' * ΔΩ`:
@inline forwarddiff_back(::Type{T}, ΔΩ, y_dual) where T = begin
    #global g_tangent = ΔΩ
    #global g_y_dual = y_dual
    shape_forwarddiff_gradient(T, forwarddiff_back_unshaped(ΔΩ, y_dual))
end

# For `x::StaticArrays.SVector`, `forwarddiff_fwd_back(f, (x,), Val(1), ΔΩ) == ForwardDiff.jacobian(f, x)' * ΔΩ`:
@inline function forwarddiff_fwd_back(f::Base.Callable, xs::Tuple, ::Val{i}, ΔΩ) where i
    # @info "RUN forwarddiff_fwd_back(f, xs, Val($i), ΔΩ)"
    x_i = xs[i]
    y_dual = forwarddiff_fwd(f, xs, Val(i))
    forwarddiff_back(typeof(x_i), ΔΩ, y_dual)
end



