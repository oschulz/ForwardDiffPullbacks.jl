# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

import Test
import ForwardDiffPullbacks
import Documenter

Test.@testset "Package ForwardDiffPullbacks" begin
    include("test_fwd_back.jl")
    include("test_rrules.jl")
    include("test_zygote.jl")

    # doctests
    Documenter.DocMeta.setdocmeta!(
        ForwardDiffPullbacks,
        :DocTestSetup,
        :(using ForwardDiffPullbacks);
        recursive=true,
    )
    Documenter.doctest(ForwardDiffPullbacks)
end # testset
