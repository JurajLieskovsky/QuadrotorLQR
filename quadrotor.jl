module Quadrotor

using LinearAlgebra
using Parameters

include("quaternions.jl")
using .Quaternions

# System's properties

struct System
    g::Vector # gravitation acceleration
    m::Real   # mass
    J::Matrix # moment of inertia
    Wrot::Matrix # moment matrix
    Wlin::Matrix # thrust matrix

    function System(gravitation_accelaration, mass, moment_of_inertia, arm_length)
        rotor_radius_vectors = [
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0]
        ]
        ϕ = pi / 32
        thrust_vectors = [
            [0, -sin(ϕ), cos(ϕ)],
            [0, sin(ϕ), cos(ϕ)],
            [0, sin(ϕ), cos(ϕ)],
            [0, -sin(ϕ), cos(ϕ)]
        ]
        moment_matrix = mapreduce(
            arg -> arg[1] × arg[2] * arm_length,
            hcat,
            zip(rotor_radius_vectors, thrust_vectors)
        )
        thrust_matrix = hcat(thrust_vectors...)

        return new(gravitation_accelaration, mass, moment_of_inertia, moment_matrix, thrust_matrix)
    end
end

# Dynamics (accelerations)

"""
Calculates the quadrotor's angular acceleration.

arguments:
    p - system's properties
    ω - angular velocity (in the frame of the quadrotor)
    u - control inputs

returns:
    ω̇ - angular accelaration

"""
function angular_acceleration(p::System, ω, u)
    @unpack J, Wrot = p
    T = Wrot * u
    return J \ (T - ω × (J * ω))
end


"""
Calculates the quadrotor's linear acceleration.

arguments:
    p - system's properties
    q - orientation of the quadrotor (quaternion)
    u - control inputs

returns:
    ω̇ - angular accelaration

"""
function linear_acceleration(p::System, q, u)
    @unpack g, m, Wlin = p
    F = Wlin*u
    return g + Quaternions.rot(q, F) / m
end

# State incrementation utility

"""
Calculates the system's state as incremented by dz.

arguments:
x₀ - system's state (x₀ = [r, q, v, ω])
dz - incrementation of the state (dz = [dr, dθ, dv, dω]) 

returns:
x₀ + dx(dz) - incremented state (dx = [dr, q̇(q,dθ), dv, dω])

"""
function incremented_state(x₀, dz)
    @assert length(x₀) == 13
    @assert length(dz) == 12

    r, q, v, ω = x₀[1:3], x₀[4:7], x₀[8:10], x₀[11:13]
    dr, dθ, dv, dω = dz[1:3], dz[4:6], dz[7:9], dz[10:12]

    return vcat(r + dr, q + Quaternions.q̇(q, dθ), v + dv, ω + dω)
end

# State difference utility

"""
Calculates the difference between the current and reference state in the tangential space of the reference state.

arguments:
    x  - current state
    x₀ - reference state

returns:
    dz - state difference (dz = [dr, dθ, dv, dω]) 
    
The approximation x ≈ x₀ + dx(dz), where dx = [dr, q̇(q,dθ), dv, dω], should be accurate for very small values of dθ
    
"""
function state_difference(x, x₀)
    @assert length(x) == 13
    @assert length(x₀) == 13

    dr = x[1:3] - x₀[1:3]
    dθ = Quaternions.dθ(Quaternions.multiply(x[4:7], Quaternions.conjugate(x₀[4:7])))
    dv = x[8:10] - x₀[8:10]
    dω = x[11:13] - x₀[11:13]

    return vcat(dr, dθ, dv, dω)
end

# State space descriptions

"""
Calculates the rate of change of the state according to the state description ẋ = f(x,u).

arguments:
x - system's state (x = [r,q,v,ω])
u - control inputs

returns:
ẋ - rate of change of the state (ẋ = [v,q̇,v̇,ω̇])

"""
function forward_dynamics(properties, x, u)
    @assert length(x) == 13
    @assert length(u) == 4

    _, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]
    ω̇ = angular_acceleration(properties, ω, u)
    v̇ = linear_acceleration(properties, q, u)

    return vcat(v, Quaternions.q̇(q, ω), v̇, ω̇)
end


"""
Calculates the rate of change of the state in the tangent direction.

arguments:
x₀ - system's state (x₀ = [r,q,v,ω])
dz - increment of the state (dz = [dr, dθ, dv, dω]) 
u  - control inputs

returns:
dż - rate of change of the state in tangent-space (dż = [v,ω,v̇,ω̇])

"""
function tangent_forward_dynamics(properties, x₀, dz, u)
    @assert length(x₀) == 13
    @assert length(dz) == 12
    @assert length(u) == 4

    x = incremented_state(x₀, dz)
    _, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]

    ω̇ = angular_acceleration(properties, ω, u)
    v̇ = linear_acceleration(properties, q, u)

    return vcat(v, ω, v̇, ω̇)
end

end
