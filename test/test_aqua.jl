# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import ForwardDiffPullbacks

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(ForwardDiffPullbacks))
end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        ForwardDiffPullbacks,
        ambiguities = false
    )
end # testset
