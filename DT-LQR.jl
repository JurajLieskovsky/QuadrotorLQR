using Revise

using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq, DiffEqCallbacks
using Plots
using MatrixEquations

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

# Control period
h = 1e-2

# Linearization
"""RK4 integration with zero-order hold on u"""
function dt_dynamics(x, u)
    f1 = QuadrotorODE.dynamics(quadrotor, x, u)
    f2 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f1, u)
    f3 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f2, u)
    f4 = QuadrotorODE.dynamics(quadrotor, x + h * f3, u)
    return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
end

fx = ForwardDiff.jacobian(x_ -> dt_dynamics(x_, u_eq), x_eq)
fu = ForwardDiff.jacobian(u_ -> dt_dynamics(x_eq, u_), u_eq)

E = QuadrotorODE.jacobian(x_eq)
A = E' * fx * E
B = E' * fu

# LQR design
Q = diagm(vcat(1e1 * ones(3), 1e1 * ones(3), ones(3), ones(3)))
R = Matrix(I(4))

S, _ = MatrixEquations.ared(A, B, R, Q)
K = inv(R + B' * S * B) * B' * S * A

controller(x) = u_eq - K * QuadrotorODE.state_difference(x, x_eq)

# Simulation
tspan = (0.0, 5.0)
θ = 3 * pi / 8
x0 = vcat(r_eq, [cos(θ / 2), sin(θ / 2), 0, 0], v_eq, ω_eq)

## Callbacks
ControllerCallback = PeriodicCallback(i -> i.p .= controller(i.u), h, initial_affect=true)
saved_values = SavedValues(Float64, Vector{Float64})
InputSavingCallback = SavingCallback((u, t, integrator) -> copy(integrator.p), saved_values)

## Problem
prob = ODEProblem(
    (x, p, _) -> QuadrotorODE.dynamics(quadrotor, x, p),
    x0,
    tspan,
    similar(u_eq)
)

## Solution
sol = solve(prob, callback=CallbackSet(ControllerCallback, InputSavingCallback))

# Plotting
Δt = 1e-2
ts = tspan[1]:Δt:tspan[2]
xs = map(t -> sol(t), ts)

state_labels = ["x" "y" "z" "q₀" "q₁" "q₂" "q₃" "vx" "vy" "vz" "ωx" "ωy" "ωz"]
input_labels = ["u₀" "u₁" "u₂" "u₃"]

plt = plot(layout=(2, 1))
plot!(
    plt, ts, mapreduce(x -> x[1:3]', vcat, xs),
    label=state_labels, subplot=1
)
plot!(
    plt, saved_values.t, mapreduce(u -> u', vcat, saved_values.saveval),
    label=input_labels, seriestype=:steppost, subplot=2
)

display(plt)

# Visualization
vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

## quadrotor and target
MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * quadrotor.a, 0.12, 0.25)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.12)

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
