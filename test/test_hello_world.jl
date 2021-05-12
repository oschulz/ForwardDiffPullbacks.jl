# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).

using ForwardDiffPullbacks
using Test


@testset "hello_world" begin
    @test ForwardDiffPullbacks.hello_world() == 42
end
