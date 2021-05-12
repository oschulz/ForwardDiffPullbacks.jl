# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ForwardDiffPullbacks" begin
    include("test_fwd_back.jl")
    include("test_rrules.jl")
    include("test_zygote.jl")
end # testset
