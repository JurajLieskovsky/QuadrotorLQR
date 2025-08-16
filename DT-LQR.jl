using Revise

using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots
using MatrixEquations

using QuadrotorODE
using MeshCatBenchmarkMechanisms

# Properties of the quadrotor
a = 0.3
quadrotor = QuadrotorODE.System([0, 0, -9.81], 1, I(3), a, 0.01)

#Quaternion stuff
function hat(v)
    return [0 -v[3] v[2];
        v[3] 0 -v[1];
        -v[2] v[1] 0]
end

function L(q)
    s = q[1]
    v = q[2:4]
    L = [s -v';
        v s*I+hat(v)]
    return L
end

T = Diagonal([1; -ones(3)])
H = [zeros(1, 3); I]
function qtoQ(q)
    return H' * T * L(q) * T * L(q) * H
end

function G(q)
    G = L(q) * H
end

function rptoq(ϕ)
    (1 / sqrt(1 + ϕ' * ϕ)) * [1; ϕ]
end

function qtorp(q)
    q[2:4] / q[1]
end

#Quadrotor parameters
m = 0.5
ℓ = 0.1750
J = Diagonal([0.0023, 0.0023, 0.004])
g = 9.81
kt = 1.0
km = 0.0245

h = 0.05 # 20 Hz

function E(q)
    E = BlockDiagonal([1.0 * I(3), G(q), 1.0 * I(6)])
end

function quad_dynamics(x, u)
    r = x[1:3]
    q = x[4:7] / norm(x[4:7]) #normalize q just to be careful
    v = x[8:10]
    ω = x[11:13]
    Q = qtoQ(q)

    ṙ = Q * v
    q̇ = 0.5 * L(q) * H * ω

    v̇ = Q' * [0; 0; -g] + (1 / m) * [zeros(2, 4); kt * ones(1, 4)] * u - hat(ω) * v

    ω̇ = J \ (-hat(ω) * J * ω + [0 ℓ*kt 0 -ℓ*kt; -ℓ*kt 0 ℓ*kt 0; km -km km -km] * u)

    return [ṙ; q̇; v̇; ω̇]
end

function quad_dynamics_rk4(x, u)
    #RK4 integration with zero-order hold on u
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5 * h * f1, u)
    f3 = quad_dynamics(x + 0.5 * h * f2, u)
    f4 = quad_dynamics(x + h * f3, u)
    xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    xn[4:7] .= xn[4:7] / norm(xn[4:7]) #re-normalize quaternion
    return xn
end

# Equlibrium
r_eq = [0, 0, 1.0]
q_eq = [1.0, 0, 0, 0]
v_eq = zeros(3)
ω_eq = zeros(3)

x_eq = vcat(r_eq, q_eq, v_eq, ω_eq)
u_eq = m * 9.81 / 4 * ones(4)

# Linearization
fx = ForwardDiff.jacobian(x_ -> quad_dynamics_rk4(x_, u_eq), x_eq)
fu = ForwardDiff.jacobian(u_ -> quad_dynamics_rk4(x_eq, u_), u_eq)

J = QuadrotorODE.motion_jacobian(x_eq)

A = J' * fx * J
B = J' * fu # Should be multiplied by 2 in my case

# LQR design
# Q = diagm(vcat(1e2 * ones(3), 1e-2 * ones(3), 1e1 * ones(3), 1e-3 * ones(3)))
# R = 1e0 * Matrix(I(4))
Q = I(12)
R = 0.1 * I(4)

S, _ = MatrixEquations.ared(A, B, R, Q)
K = inv(R + B' * S * B) * B' * S * A

controller(x) = u_eq - K * QuadrotorODE.state_difference(x, x_eq)

#=
# Simulation
tspan = (0.0, 10.0)
θ = 3 * pi / 8
x0 = vcat([0, 0, 0], [cos(θ / 2), sin(θ / 2), 0, 0], v_eq, ω_eq)

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

=#
