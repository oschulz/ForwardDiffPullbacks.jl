# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


# ToDo: Use ProjectTo in rrules


struct FwdDiffPullbackThunk{F<:Function,T<:Tuple,i,U<:Any} <: ChainRulesCore.AbstractThunk
    f::F
    xs::T
    ΔΩ::U
end

function FwdDiffPullbackThunk(f::F, xs::T, ::Val{i}, ΔΩ::U) where {F<:Function,T<:Tuple,i,U<:Any}
    FwdDiffPullbackThunk{F,T,i,U}(f, xs, ΔΩ)
end

@inline function ChainRulesCore.unthunk(tnk::FwdDiffPullbackThunk{F,T,i,U}) where {F,T,i,U}
    forwarddiff_vjp_impl(tnk.f, tnk.xs, Val(i), unthunk(tnk.ΔΩ))
end


Base.@generated function _forwarddiff_pullback_thunks(f::Function, xs::NTuple{N,Any}, ΔΩ::Any) where N
    Expr(:tuple, NoTangent(), (:(ForwardDiffPullbacks.FwdDiffPullbackThunk(f, xs, Val($i), ΔΩ)) for i in 1:N)...)
end

g_ΔΩ = nothing

@inline function ChainRulesCore.rrule(wrapped_f::WithForwardDiff, xs::Vararg{Any,N}) where N
    # @info "RUN ChainRulesCore.rrule(wrapped_f::WithForwardDiff, xs) with N = $N"
    f = wrapped_f.f
    y = f(xs...)
    with_fwddiff_pullback(ΔΩ) = begin
        #@info "XXXXXXXXXX" ΔΩ
        #global g_ΔΩ = ΔΩ
        _forwarddiff_pullback_thunks(f, xs, ΔΩ)
    end
    return y, with_fwddiff_pullback
end



struct FwdDiffBCPullbackThunk{F<:Function,T<:Tuple,i,U} <: ChainRulesCore.AbstractThunk
    f::F
    Xs::T
    ΔΩA::U
end

function FwdDiffBCPullbackThunk(f::F, Xs::T, ::Val{i}, ΔΩA::U) where {F<:Function,T<:Tuple,i,U}
    FwdDiffBCPullbackThunk{F,T,i,U}(f, Xs, ΔΩA)
end

@inline function ChainRulesCore.unthunk(tnk::FwdDiffBCPullbackThunk{F,T,i,U}) where {F,T,i,U}
    r = forwarddiff_bc_vjp_impl(tnk.f, tnk.Xs, Val(i), unthunk(tnk.ΔΩA))
    # @info "ChainRulesCore.unthunk returns" tnk.Xs tnk.ΔΩA r
    r
end



Base.@generated function _forwarddiff_bc_pullback_thunks(f::Function, Xs::NTuple{N,Any}, ΔΩA::Any) where N
    Expr(:tuple, NoTangent(), NoTangent(), (:(ForwardDiffPullbacks.FwdDiffBCPullbackThunk(f, Xs, Val($i), ΔΩA)) for i in 1:N)...)
end

function ChainRulesCore.rrule(::typeof(Base.broadcasted), wrapped_f::WithForwardDiff, Xs::Vararg{Any,N}) where N
    # @info "RUN ChainRulesCore.rrule(Base.broadcasted, wrapped_f::WithForwardDiff, Xs) with N = $N"
    f = wrapped_f.f
    y = broadcast(f, Xs...)
    bc_with_fwddiff_pullback(ΔΩA) = _forwarddiff_bc_pullback_thunks(f, Xs, ΔΩA)
    return y, bc_with_fwddiff_pullback
end
