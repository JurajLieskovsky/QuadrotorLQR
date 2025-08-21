using Revise

using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using DiffEqCallbacks, NonlinearSolve
using Plots
using MatrixEquations
using StaticArrays

using QuadrotorODE
using MeshCatBenchmarkMechanisms

# Properties of the quadrotor
quadrotor = QuadrotorODE.System(9.81, 0.5, diagm([0.0023, 0.0023, 0.004]), 0.1750, 1.0, 0.0245)

# Equlibrium
r_eq = [0, 0, 1.0]
q_eq = [1.0, 0, 0, 0]
v_eq = zeros(3)
ω_eq = zeros(3)

x_eq = vcat(r_eq, q_eq, v_eq, ω_eq)
u_eq = quadrotor.m * quadrotor.g / 4 * ones(4)

# Linearization
fx = ForwardDiff.jacobian(x_ -> QuadrotorODE.dynamics(quadrotor, x_, u_eq), x_eq)
fu = ForwardDiff.jacobian(u_ -> QuadrotorODE.dynamics(quadrotor, x_eq, u_), u_eq)

E = QuadrotorODE.jacobian(x_eq)

A = E' * fx * E
B = E' * fu

# LQR design
Q = diagm(vcat(1e1 * ones(3), 1e1 * ones(3), ones(3), ones(3)))
R = Matrix(I(4))

S, _ = MatrixEquations.arec(A, B, R, Q)
K = inv(R) * B' * S

function controller(x)
    u = u_eq - K * QuadrotorODE.state_difference(x, x_eq)
    return map(u_k -> u_k >= 0 ? u_k : 0, u)
end

# Simulation
tspan = (0.0, 5.0)
θ = 3 * pi / 4
x0 = vcat(r_eq, [cos(θ / 2), sin(θ / 2), 0, 0], v_eq, ω_eq)

UnitQuatProjectionCallback = DiffEqCallbacks.ManifoldProjection(
    (x, _, _) -> [x[4:7]' * x[4:7] - 1], autodiff=AutoForwardDiff()
)

prob = ODEProblem(
    (x, _, _) -> QuadrotorODE.dynamics(quadrotor, x, controller(x)),
    x0,
    tspan
)
sol = solve(prob, callback=UnitQuatProjectionCallback)

# Plotting
Δt = 1e-2
ts = tspan[1]:Δt:tspan[2]
xs = map(t -> sol(t), ts)
us = map(x -> controller(x), xs)

state_labels = ["x" "y" "z" "q₀" "q₁" "q₂" "q₃" "vx" "vy" "vz" "ωx" "ωy" "ωz"]
input_labels = ["u₀" "u₁" "u₂" "u₃"]

plt = plot(layout=(2, 1))
plot!(plt, ts, mapreduce(x -> x[1:3]', vcat, xs), label=state_labels, subplot=1)
plot!(plt, ts, mapreduce(u -> u', vcat, us), label=input_labels, subplot=2)

display(plt)

# Visualization
vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

## quadrotor and target
MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * quadrotor.a, 0.07, 0.12)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.07)

## initial configuration
MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, xs[1])
MeshCatBenchmarkMechanisms.set_target_position!(vis, x_eq[1:3])

## animation
anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=1 / Δt)
for (i, x) in enumerate(xs)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, x)
    end
end
setanimation!(vis, anim, play=false);
