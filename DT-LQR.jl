using Revise

using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots
using MatrixEquations

using QuadrotorODE
using MeshCatBenchmarkMechanisms

# Properties of the quadrotor
a = 0.1750
quadrotor = QuadrotorODE.System([0, 0, -9.81], 0.5, Diagonal([0.0023, 0.0023, 0.004]), a, 1.0, 0.0245)

function quad_dynamics_rk4(x, u)
    #RK4 integration with zero-order hold on u
    h = 0.05
    f1 = QuadrotorODE.dynamics(quadrotor, x, u)
    f2 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f1, u)
    f3 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f2, u)
    f4 = QuadrotorODE.dynamics(quadrotor, x + h * f3, u)
    xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return xn
end

# Equlibrium
r_eq = [0, 0, 1.0]
q_eq = [1.0, 0, 0, 0]
v_eq = zeros(3)
ω_eq = zeros(3)

x_eq = vcat(r_eq, q_eq, v_eq, ω_eq)
u_eq = quadrotor.m * 9.81 / 4 * ones(4)

# Linearization
fx = ForwardDiff.jacobian(x_ -> quad_dynamics_rk4(x_, u_eq), x_eq)
fu = ForwardDiff.jacobian(u_ -> quad_dynamics_rk4(x_eq, u_), u_eq)

j = QuadrotorODE.jacobian(x_eq)
A = j' * fx * j
B = j' * fu # Should be multiplied by 2 in my case

# LQR design
# Q = diagm(vcat(1e2 * ones(3), 1e-2 * ones(3), 1e1 * ones(3), 1e-3 * ones(3)))
# R = 1e0 * Matrix(I(4))
Q = I(12)
R = 0.1 * I(4)

S, _ = MatrixEquations.ared(A, B, R, Q)
K = inv(R + B' * S * B) * B' * S * A

controller(x) = u_eq - K * QuadrotorODE.state_difference(x, x_eq)

# Simulation
tspan = (0.0, 4.0)
θ = 3 * pi / 8
x0 = vcat([0, 0, 1.0], [cos(θ / 2), sin(θ / 2), 0, 0], v_eq, ω_eq)

prob = ODEProblem(
    (x, _, _) -> QuadrotorODE.dynamics(quadrotor, x, controller(x)),
    x0,
    tspan
)
sol = solve(prob)

# Plotting
ts = tspan[1]:1e-2:tspan[2]
xs = map(t -> sol(t), ts)
us = map(x -> controller(x), xs)

state_labels = ["x" "y" "z" "q₀" "q₁" "q₂" "q₃" "vx" "vy" "vz" "ωx" "ωy" "ωz"]
input_labels = ["u₀" "u₁" "u₂" "u₃"]

plt = plot(layout=(2, 1))
plot!(plt, ts, mapreduce(x -> x', vcat, xs), label=state_labels, subplot=1)
plot!(plt, ts, mapreduce(u -> u', vcat, us), label=input_labels, subplot=2)

display(plt)

# Visualization
vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

## quadrotor and target
MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * a, 0.12, 0.25)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.12)

## initial configuration
MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, xs[1])
MeshCatBenchmarkMechanisms.set_target_position!(vis, x_eq[1:3])

## animation
anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=1 / 1e-2)
for (i, x) in enumerate(xs)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, x)
    end
end
setanimation!(vis, anim, play=false);
