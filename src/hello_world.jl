# This file is a part of ForwardDiffPullbacks.jl, licensed under the MIT License (MIT).


"""
    ForwardDiffPullbacks.hello_world()

Prints "Hello, World!" and returns 42.

```jldoctest
using ForwardDiffPullbacks

ForwardDiffPullbacks.hello_world()

# output

Hello, World!
42
```
"""
function hello_world()
    println("Hello, World!")
    return 42
end
