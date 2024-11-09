using Revise
using LinearAlgebra
using ForwardDiff

include("quadrotor.jl")
using .Quadrotor

# Properties of the quadrotor
quadrotor = Quadrotor.System([0, 0, -9.81], 1, I(3), 0.3)

# Equlibrium
x₀ = zeros(3)
q₀ = [1, 0, 0, 0]
v₀ = zeros(3)
ω₀ = zeros(3)

x₀ = vcat(x₀, q₀, v₀, ω₀)

u₀ = 9.81 / 4 * ones(4)

## Equilibrium double-check
ω̇₀ = Quadrotor.angular_acceleration(quadrotor, [0, 0, 0], u₀)
v̇₀ = Quadrotor.linear_acceleration(quadrotor, q₀, u₀)

@assert ω̇₀ == zeros(3)
@assert v̇₀ == zeros(3)

# Linearization of the system's dynamics (tangent-space)
dz₀ = zeros(12)
A = ForwardDiff.jacobian(dz -> Quadrotor.tangent_forward_dynamics(quadrotor, x₀, dz, u₀), dz₀)
B = ForwardDiff.jacobian(u -> Quadrotor.tangent_forward_dynamics(quadrotor, x₀, dz₀, u), u₀)
