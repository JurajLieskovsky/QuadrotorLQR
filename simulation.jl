using Revise

using LinearAlgebra
using ForwardDiff
using MatrixEquations: arec
using OrdinaryDiffEq
using Plots

include("quadrotor.jl")
using .Quadrotor

# Properties of the quadrotor
quadrotor = Quadrotor.System([0, 0, -9.81], 1, I(3), 0.3)

# Equlibrium
r₀ = zeros(3)
q₀ = [1, 0, 0, 0]
v₀ = zeros(3)
ω₀ = zeros(3)

x₀ = vcat(r₀, q₀, v₀, ω₀)

u₀ = 9.81 / 4 * ones(4) / cos(pi / 32)

## validation
ω̇₀ = Quadrotor.angular_acceleration(quadrotor, [0, 0, 0], u₀)
v̇₀ = Quadrotor.linear_acceleration(quadrotor, q₀, u₀)

@assert ω̇₀ == zeros(3)
@assert v̇₀ == zeros(3)

# LQR controller
## linearization of the system's dynamics (tangent-space)
dz₀ = zeros(12)
A = ForwardDiff.jacobian(dz -> Quadrotor.tangent_forward_dynamics(quadrotor, x₀, dz, u₀), dz₀)
B = ForwardDiff.jacobian(u -> Quadrotor.tangent_forward_dynamics(quadrotor, x₀, dz₀, u), u₀)

## running cost
Q = diagm(vcat(1e1 * ones(6), 1e0 * ones(6)))
R = 1e-1 * Matrix(I(4))

## state-feedback
P, _ = arec(A, B, R, Q)
K = -inv(R) * B' * P

# Simulation
function controlled_dynamics(quadrotor, x₀, u₀, x)
    dx = Quadrotor.state_difference(x, x₀)
    du = K * dx
    Quadrotor.forward_dynamics(quadrotor, x, u₀ + du)
end

prob = ODEProblem(
    (x, _, _) -> controlled_dynamics(quadrotor, x₀, u₀, x),
    vcat([-1, -1, -1], q₀, v₀, ω₀),
    (0.0, 5.0)
)
sol = solve(prob)

# Plotting
plt = plot()
plot!(plt, sol, idxs=[1,2,3], label=["x" "y" "z"])
plot!(plt, sol, idxs=[4,5,6,7], label=["q₀" "q₁" "q₂" "q₃"])
# plot!(plt, sol, idxs=[8,9,10], label=["vx" "vy" "vz"])
# plot!(plt, sol, idxs=[11,12,13], label=["ωx" "ωy" "ωz"])
