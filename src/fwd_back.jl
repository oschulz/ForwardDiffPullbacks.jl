# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = dual_number(TagType, x, (true,))

function forwarddiff_dualized(::Type{TagType}, x::NTuple{N,Real}) where {TagType,N}
    ntuple(j -> dual_number(TagType, x[j], ntuple(i -> i == j, Val(N))), Val(N))
end

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::SVector) where {TagType} = SVector(forwarddiff_dualized(TagType, (x...,)))
forwarddiff_dualized(::Type{TagType}, x::SVector{0}) where {TagType} = x

forwarddiff_dualized(::Type{TagType}, x) where TagType = unflatten(x, forwarddiff_dualized(TagType, flatten(x)))


@inline tangent_dual_product(ΔΩ::RealTangentLike, y_dual::Real) = ΔΩ * dual_partials(y_dual)
@inline tangent_dual_product(ΔΩ_i::ZeroLike, y_dual::Any) = ZeroTangent()

tangent_dual_product(ΔΩ::Tuple{}, y_dual::Tuple{}) = NoTangent()

tangent_dual_product(ΔΩ::Tuple, y_dual::Tuple) = sum(map(tangent_dual_product, ΔΩ, y_dual))

function tangent_dual_product(ΔΩ::NTuple{N,RealTangentLike}, y_dual::NTuple{N,RealDualLike}) where N
    sum(map(tangent_dual_product, ΔΩ, y_dual))
end

@inline function tangent_dual_product(ΔΩ::NamedTuple{names}, y_dual::NamedTuple{names}) where {names}
    tangent_dual_product(values(ΔΩ), values(y_dual))
end

@inline function tangent_dual_product(ΔΩ::SVector{N}, y_dual::SVector{N}) where N
    tangent_dual_product((ΔΩ...,), (y_dual...,))
end

function tangent_dual_product(ΔΩ::NamedTuple{names}, y_dual::Any) where {names}
    tangent_dual_product(values(ΔΩ), getprops_by_name(y_dual, Val(names)))
end

function tangent_dual_product(ΔΩ::Tangent, y_dual::Any)
    stripped_ΔΩ = ChainRulesCore.backing(ChainRulesCore.canonicalize(ΔΩ))
    tangent_dual_product(stripped_ΔΩ, y_dual)
end

svec_tangent_dual_product(ΔΩ, y_dual) = make_svector((tangent_dual_product(ΔΩ, y_dual)...,))


@inline function forwarddiff_fwd(f::F, xs::Tuple, ::Val{i}) where {F<:Function,i}
    TagType = dual_tagtype(f, Val(i), eltype(xs[i]))
    xs_i_dual = forwarddiff_dualized(TagType, xs[i])
    xs_dual = ntuple(j -> i == j ? xs_i_dual : xs[j], Val(length(xs)))
    f(xs_dual...)
end


# For `x::StaticArrays.SVector`, `forwarddiff_vjp_impl(f, (x,), Val(1), ΔΩ) == ForwardDiff.jacobian(f, x)' * ΔΩ`:
@inline function forwarddiff_vjp_impl(f::F, xs::Tuple, ::Val{i}, ΔΩ) where {F<:Function,i}
    #@info "RUN forwarddiff_vjp_impl(f, $xs, $i, $ΔΩ)"
    x_i = xs[i]
    y_dual = forwarddiff_fwd(f, xs, Val(i))
    unflatten_tangent(x_i, svec_tangent_dual_product(ΔΩ, y_dual))
end


# Similar to Zygote.unbroadcast:

unflatten_bc_tangent(x_orig::Number, dx_flat::AbstractArray{<:Real}) =
    unflatten_tangent(x_orig, dx_flat)

unflatten_bc_tangent(x_orig::Tuple{<:Any}, dx_flat::AbstractArray{<:Real}) =
    make_tangent(typeof(x_orig), (unflatten_tangent(first(x_orig), dx_flat),))

unflatten_bc_tangent(x_orig::Base.RefValue, dx_flat::AbstractArray{<:Real}) =
    make_tangent(typeof(x_orig), (x = unflatten_tangent(first(x_orig), dx_flat),))

unflatten_bc_tangent(x_orig::Number, dx_flat::AbstractArray{<:AbstractArray{<:Real}}) =
    unflatten_tangent(x_orig, sum(dx_flat))

unflatten_bc_tangent(x_orig::Tuple{<:Any}, dx_flat::AbstractArray{<:AbstractArray{<:Real}}) =
    make_tangent(typeof(x_orig), (unflatten_tangent(first(x_orig), sum(dx_flat)),))

unflatten_bc_tangent(x_orig::Base.RefValue, dx_flat::AbstractArray{<:AbstractArray{<:Real}}) =
    make_tangent(typeof(x_orig), (x = unflatten_tangent(first(x_orig), sum(dx_flat)),))

unflatten_bc_tangent(x_orig::Tuple, dx_flat::AbstractArray{<:AbstractArray{<:Real}}) =
    make_tangent(typeof(x_orig), (unflatten_tangent.(x_orig, sum(dx_flat, dims=2:ndims(dx_flat)))...,))

function unflatten_bc_tangent(x_orig::AbstractArray, dx_flat::AbstractArray{<:AbstractArray{<:Real}})
    dims = ntuple(d -> size(x_orig, d) == 1 ? d : ndims(dx_flat)+1, ndims(dx_flat))
    unflatten_tangent.(x_orig, sum(dx_flat; dims = dims))
end
    
unflatten_bc_tangent(x_orig::Any, dx_flat::AbstractArray{<:ChainRulesCore.AbstractZero}) =
    ConstructionBase.constructorof(eltype(dx_flat))()


struct FwddiffVJPSingleArg{F<:Function,i} <: Function
    f::F
end
FwddiffVJPSingleArg(f::F, ::Val{i}) where {F<:Function,i} = FwddiffVJPSingleArg{F,i}(f)

# Need to use Vararg{Any,N} to force specialization:
function (fwdback::FwddiffVJPSingleArg{F,i})(ΔΩ, xs::Vararg{Any,N}) where {F,i,N}
    y_dual = forwarddiff_fwd(fwdback.f, xs, Val(i))
    svec_tangent_dual_product(ΔΩ, y_dual)
end


function forwarddiff_bc_vjp_impl(f::F, Xs::Tuple, ::Val{i}, ΔΩA::Any) where {F<:Function,i}
    # @info "RUN forwarddiff_bc_vjp_impl(f, Xs, Val($i), ΔΩA)"
    vjp_i = FwddiffVJPSingleArg(f, Val(i))
    dXi_flat = broadcast(vjp_i, ΔΩA, Xs...) # ToDo: Use Base.Broadcast.broadcasted
    unflatten_bc_tangent(Xs[i], dXi_flat)
end
