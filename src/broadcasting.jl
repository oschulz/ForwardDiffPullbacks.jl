# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


struct FwddiffFwd{F<:Base.Callable,i} <: Function
    f::F
end
FwddiffFwd(f::F, ::Val{i}) where {F<:Base.Callable,i} = FwddiffFwd{F,i}(f)

(fwd::FwddiffFwd{F,i})(xs...) where {F,i} = forwarddiff_fwd(fwd.f, xs, Val(i))


struct FwddiffBack{TX<:Any} <: Function end

(bck::FwddiffBack{TX})(ΔΩ, y_dual) where TX = forwarddiff_back(TX, ΔΩ, y_dual)


function forwarddiff_bc_fwd_back(f::Base.Callable, Xs::Tuple, ::Val{i}, ΔΩA::Any) where i
    # @info "RUN forwarddiff_bc_fwd_back(f, Xs, Val($i), ΔΩA)"
    fwd = FwddiffFwd(f, Val(i))
    TX = eltype(Xs[i])
    bck = FwddiffBack{TX}()
    bck.(ΔΩA, fwd.(Xs...))
end
