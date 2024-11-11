using Revise

using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots

using QuadrotorODE

using MeshCat
using CoordinateTransformations
using Rotations
using GeometryBasics
using Colors: RGB

lqr_time_domain = :continuous

if lqr_time_domain == :continuous
    using MatrixEquations: arec

elseif lqr_time_domain == :discrete
    using RungeKutta
    using MatrixEquations: ared
end


# Properties of the quadrotor
a = 0.3
quadrotor = QuadrotorODE.System([0, 0, -9.81], 1, I(3), a, 0.01)

# Equlibrium
r₀ = zeros(3)
q₀ = [1, 0, 0, 0]
v₀ = zeros(3)
ω₀ = zeros(3)

x₀ = vcat(r₀, q₀, v₀, ω₀)
u₀ = 9.81 / 4 * ones(4)

## validation
ω̇₀ = QuadrotorODE.angular_acceleration(quadrotor, [0, 0, 0], u₀)
v̇₀ = QuadrotorODE.linear_acceleration(quadrotor, q₀, u₀)

@assert ω̇₀ == zeros(3)
@assert v̇₀ == zeros(3)

# LQR controller
## linearization of the system's dynamics (tangent-space)
if lqr_time_domain == :continuous
    dz₀ = zeros(12)
    A = ForwardDiff.jacobian(dz -> QuadrotorODE.tangential_forward_dynamics(quadrotor, x₀, dz, u₀), dz₀)
    B = ForwardDiff.jacobian(u -> QuadrotorODE.tangential_forward_dynamics(quadrotor, x₀, dz₀, u), u₀)

elseif lqr_time_domain == :discrete
    rk4 = RungeKutta.RK4()

    f!(dznew, x₀, dz, u) = RungeKutta.f!(
        dznew,
        rk4,
        (dznew, dz, u) -> dznew .= QuadrotorODE.tangential_forward_dynamics(quadrotor, x₀, dz, u),
        dz,
        u,
        1e-4
    )

    dz₀ = zeros(12)
    A = ForwardDiff.jacobian((dznew, dz) -> f!(dznew, x₀, dz, u₀), zeros(12), dz₀)
    B = ForwardDiff.jacobian((dznew, u) -> f!(dznew, x₀, dz₀, u), zeros(12), u₀)
end

## running cost
Q = diagm(vcat(1e1 * ones(6), 1e0 * ones(6)))
R = 1e0 * Matrix(I(4))

## state-feedback
if lqr_time_domain == :continuous
    P, _ = arec(A, B, R, Q)

elseif lqr_time_domain == :discrete
    P, _ = ared(A, B, R, Q)
end

K = -inv(R) * B' * P

## controller
controller(x₀, u₀, x) = u₀ + K * QuadrotorODE.state_difference(x, x₀)

# Simulation
tspan = (0.0, 10.0)
x0 = vcat([5, -5, -1], [1, 0, 0, 0], v₀, ω₀)
prob = ODEProblem(
    (x, _, _) -> QuadrotorODE.forward_dynamics(quadrotor, x, controller(x₀, u₀, x)),
    x0,
    tspan
)
sol = solve(prob)

# Plotting
## states
state_plot = plot()
plot!(state_plot, sol, idxs=1:3, label=["x" "y" "z"])
plot!(state_plot, sol, idxs=4:7, label=["q₀" "q₁" "q₂" "q₃"])
# plot!(plt, sol, idxs=8:10, label=["vx" "vy" "vz"])
# plot!(plt, sol, idxs=11:13, label=["ωx" "ωy" "ωz"])

## inputs
ts = tspan[1]:1e-3:tspan[2]
us = mapreduce(x -> controller(x₀, u₀, x)', vcat, sol.(ts))
input_plot = plot(ts, us, xlabel="t", label=["u₀" "u₁" "u₂" "u₃"])

## combined
plot(state_plot, input_plot, layout=(2, 1))

# Visualization
vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

## materials (colors)
red = MeshPhongMaterial(color=RGB(1, 0, 0))
green = MeshPhongMaterial(color=RGB(0, 1, 0))
blue = MeshPhongMaterial(color=RGB(0, 0, 1))

## quadrotor
setobject!(vis[:quadrotor][:body], HyperRectangle(Vec(-a, -a, -0.06), Vec(2 * a, 2 * a, 0.12)), green)

for (i, (x, y)) in enumerate([[a, -a], [a, a], [-a, a], [-a, -a]])
    setobject!(vis[:quadrotor][Symbol("prop$i")], Cylinder(Point(x, y, 0.03), Point(x, y, 0.09), 0.25), blue)
end

## target position
setobject!(vis[:target], Sphere(Point(0.0, 0, 0), 0.12), red)

## initial configuration
x0 = sol.u[1]
settransform!(vis[:quadrotor], Translation(x0[1:3]) ∘ LinearMap(QuatRotation(x0[4:7])))

## animation
timestamps = tspan[1]:1e-2:tspan[2]
anim = MeshCat.Animation(vis, fps=100)
for (i, t) in enumerate(timestamps[2:end])
    atframe(anim, i) do
        x = sol(t)
        settransform!(vis[:quadrotor], Translation(x[1:3]) ∘ LinearMap(QuatRotation(x[4:7])))
    end
end
setanimation!(vis, anim, play=false);
