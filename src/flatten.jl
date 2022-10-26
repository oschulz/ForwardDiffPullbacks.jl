# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

_fieldvals(x::Tuple) = x

@generated function _fieldvals(x)
    accessors = [:(getfield(x, $i)) for i in 1:fieldcount(x)]
    :(($(accessors...),))
end


getndof(x::Real) = static(1)
getndof(x::Tuple{}) = static(0)
getndof(x::NTuple{N,Real}) where N = static(N)
getndof(x::T) where {T} = sum(map(getndof, _fieldvals(x)))
getndof(x::StaticArrays.StaticArray{sz, <:Real}) where sz = static(prod(size(x)))
getndof(x::AbstractArray{<:Real}) = prod(size(x))
# getndof(x::AbstractArray) # ToDo


struct StaticUnitRange{from,to} <: AbstractUnitRange{Int} end
Base.@pure StaticUnitRange(from::Int, to::Int) = StaticUnitRange{from,to}()
Base.first(::StaticUnitRange{from,to}) where {from,to} = static(from)
Base.last(::StaticUnitRange{from,to}) where {from,to} = static(to)

_getpart(A::AbstractVector{<:Real}, idxs::AbstractUnitRange{Int}) = view(A, idxs)
@generated function _getpart(A::StaticArrays.StaticVector{N,<:Real}, ::StaticUnitRange{from,to}) where {N,from,to}
    :(make_svector(($([:(A[$i]) for i in from:to]...),)))
end

_first(A) = first(A)
_first(::StaticArrays.SOneTo) = static(1)
_first(::Base.OneTo) = static(1)
_firstidx(A::AbstractArray) = _first(eachindex(IndexLinear(), A))

_make_range(from::Integer, to::Integer) = from:to
_make_range(::StaticInt{from}, ::StaticInt{to}) where {from,to} = StaticUnitRange(from, to)


@inline @generated function _partwise(f::Function, x_orig::Tuple, x_flat::AbstractVector{<:Real})
    last_idxs = Symbol("idxs_0")
    result = quote
        i0 = _firstidx(x_flat)
        $last_idxs=_make_range(i0, i0-static(1))
    end
    for i in 1:fieldcount(x_orig)
        idxs = Symbol("idxs_$i")
        expr = quote
            $idxs = _make_range(last($last_idxs) + static(1), last($last_idxs) + getndof(getfield(x_orig, $i)))
        end
        append!(result.args, expr.args)
        last_idxs = idxs
    end
    getpart_exprs = [:(f(getfield(x_orig, $i), _getpart(x_flat, $(Symbol("idxs_$i"))))) for i in 1:fieldcount(x_orig)]
    push!(result.args, :(($(getpart_exprs...),)))
    result
end


make_svector(x::Tuple{}) = SVector{0,Bool}()
make_svector(x::Tuple) = SVector(x)
make_svector(x::Tuple{ChainRulesCore.AbstractZero}) = first(x)

@inline reconstruct(::Type{T}, fieldvals...) where T = ConstructionBase.constructorof(T)(fieldvals...)
@inline reconstruct(::Type{<:NamedTuple{names}}, fieldvals...) where names = NamedTuple{names}(fieldvals)

flatten(x::Real) = SVector(x)
flatten(x::NTuple{N,Real}) where N = make_svector(x)
flatten(x::Tuple) = vcat(map(flatten, x)...)
flatten(x::AbstractArray{<:Real}) = vec(x)
flatten(x::AbstractArray) = mapreduce(flatten, vcat, x)
flatten(x::T) where T = flatten(_fieldvals(x))

unflatten(x_orig::Real, x_flat::StaticArrays.StaticVector{1,<:Real}) = first(x_flat)
function unflatten(x_orig::Real, x_flat::AbstractArray{<:Real})
    length(eachindex(x_flat)) == 1 || throw(DimensionMismatch("Cannot unflatten real values from vectors with length unequal 1."))
    first(x_flat)
end
unflatten(x_orig::NTuple{N,Real}, x_flat::AbstractArray{<:Real}) where N = (x_flat...,)
unflatten(x_orig::Tuple, x_flat::AbstractArray{<:Real}) = _partwise(unflatten, x_orig, x_flat)
unflatten(x_orig::AbstractArray{<:Real}, x_flat::AbstractArray{<:Real}) = reshape(x_flat, size(x_orig))
# unflatten(x_orig::AbstractArray, x_flat::AbstractArray{<:Real}) = ... # ToDo
unflatten(x_orig::T, x_flat::AbstractArray{<:Real}) where T = reconstruct(T, unflatten(_fieldvals(x_orig), x_flat)...)

# Required for implementation of custom_test_rrule in tests:
unflatten(x_orig::Tangent, x_flat::AbstractArray{<:Real}) = typeof(x_orig)(unflatten(ForwardDiffPullbacks._fieldvals(x_orig), x_flat)...)

unflatten_tangent(x_orig::Real, dx_flat::AbstractArray{<:Real}) = unflatten(x_orig, dx_flat)
unflatten_tangent(x_orig::NTuple{N,Real}, dx_flat::AbstractArray{<:Real}) where N = make_tangent(typeof(x_orig), unflatten(x_orig, dx_flat))
unflatten_tangent(x_orig::Tuple, dx_flat::AbstractArray{<:Real}) = make_tangent(typeof(x_orig), _partwise(unflatten_tangent, x_orig, dx_flat))
unflatten_tangent(x_orig::AbstractArray{<:Real}, dx_flat::AbstractArray{<:Real}) = unflatten(x_orig, dx_flat)
# unflatten_tangent(x_orig::AbstractArray, dx_flat::AbstractArray{<:Real}) = ... # ToDo
unflatten_tangent(x_orig::T, dx_flat::AbstractArray{<:Real}) where T = make_tangent(T, NamedTuple{fieldnames(T)}(_partwise(unflatten_tangent, _fieldvals(x_orig), dx_flat)))
unflatten_tangent(x_orig::T, dx_flat::ChainRulesCore.AbstractZero) where T = dx_flat


@generated function getprops_by_name(x, ::Val{names}) where {names}
    tpl = :(())
    append!(tpl.args, [:((x.$n)) for n in names])
    tpl
end
