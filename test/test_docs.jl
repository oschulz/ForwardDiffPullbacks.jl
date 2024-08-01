# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using Test
using ForwardDiffPullbacks
import Documenter

Documenter.DocMeta.setdocmeta!(
    ForwardDiffPullbacks,
    :DocTestSetup,
    :(using ForwardDiffPullbacks);
    recursive=true,
)
Documenter.doctest(ForwardDiffPullbacks)
