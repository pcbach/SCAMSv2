include("Arnoldi/ArnoldiMethodMod.jl")
using .ArnoldiMethodMod
using LinearAlgebra
using SparseArrays
using Plots
using Distributions
using LaTeXStrings

function ∇f(x, α)
    ans = zeros(length(x))
    ans[real.(x).<=1 ./ (4 .* real.(α) .^ 2)] = α[real.(x).<=1 ./ (4 .* real.(α) .^ 2)]
    ans[real.(x).>=1 ./ (4 .* real.(α) .^ 2)] = (1 / 2) * (x[real.(x).>=1 ./ (4 .* real.(α) .^ 2)] .^ (-1 / 2))
    return real.(ans)
end

function f(x)
    return real.(sum(sqrt.(x)))
end

function LineSearch(x, q; ε=1e-8)
    b = 0
    e = 1
    while e - b > ε
        mid1 = b + (e - b) / 3
        mid2 = e - (e - b) / 3
        vmid1 = (1 - mid1) * x + mid1 * q
        vmid2 = (1 - mid2) * x + mid2 * q
        if f(vmid1) < f(vmid2)
            b = mid1
        else
            e = mid2
        end
    end
    return (e + b) / 2
end


function powerIteration(C, y; maxIter=1000000, tol=1e-3)
    n = size(C)[1]
    u = rand(n)
    u = u / norm(u)
    for i = 1:maxIter
        u = sqrt.(y) .* (C * (sqrt.(y) .* u))
        u = u / norm(u)
        #eigenval
        val = u' * (sqrt.(y) .* (C * (sqrt.(y) .* u)))
        if norm(sqrt.(y) .* (C * (sqrt.(y) .* u)) - val * u) < (abs(tol) * abs(val))
            break
        end
    end
    v = C * (sqrt.(y) .* u)
    scale = u' * (sqrt.(y) .* v)
    v = v / sqrt(scale)
    return v, abs.(v .^ 2), u' * (sqrt.(y) .* (C * (sqrt.(y) .* u)))
end

function FullEig(C, y)
    Cbar = diagm(sqrt.(y)) * C * diagm(sqrt.(y))
    eig, eigv = eigen(Cbar)
    u = real(eigv[:, end])
    v = C * (sqrt.(y) .* u)
    scale = u' * (sqrt.(y) .* v)
    v = v / sqrt(scale)
    return v, abs.(v .^ 2), eig[end]
end

function ArnoldiGrad(C, y; tol=1e-2)
    decomp, history = partialschur(C, y, tol=tol, which=LM())
    eig, eigv = partialeigen(decomp)
    u = real(eigv[:, end])
    v = C * (sqrt.(y) .* u)
    scale = u' * (sqrt.(y) .* v)
    v = v / sqrt(scale)
    return v, abs.(v .^ 2), eig[end]
end

function LMO(C, y; maxIter=100000, tol=1e-3, eigSolver="A")
    if (eigSolver == "A")
        return ArnoldiGrad(C, y; tol=tol)
    elseif (eigSolver == "P")
        return powerIteration(C, y, maxIter=maxIter, tol=tol)
    else
        return FullEig(C, y)
    end
end

function SCAMS(C, x, z; ε=1e-2, printOption="full", eigSolver="A")
    n = length(x)
    t = 1

    tlog = Int64[]
    xlog = Vector{ComplexF64}[] 
    vlog = Vector{ComplexF64}[]
    qlog = Vector{ComplexF64}[]
    λlog = ComplexF64[]
    flog = Float64[]
    gaplog = Float64[]
    α = sqrt(2 * tr(C)) ./ (2 * diag(C))
    α = Vector(α)
    v, q, λ = LMO(C, ∇f(x, α), tol=1.0, eigSolver=eigSolver)
    δ = abs(dot(∇f(x, α), real.(q - x))) / abs(f(x))
    while t <= 1 || abs(dot(∇f(x, α), real.(q - x))) / abs(f(x)) > ε
        if t > 10
            γ = LineSearch(x, q)
        else
            γ = 2 / (t + 5)
        end
        x = (1 - γ) * x + γ * q

        for i = 1:size(z)[2]
            z[:, i] = sqrt(1 - γ) * z[:, i] + sqrt(γ) * v * rand(Normal(0, 1))
        end
        v, q, λ = LMO(C, ∇f(x, α), tol=δ / 10, eigSolver=eigSolver)
        δ = abs(dot(∇f(x, α), real.(q - x))) / abs(f(x))
        if ispow2(t)
            append!(tlog, t)
            push!(xlog, x)
            push!(vlog, v)
            push!(qlog, q)
            append!(λlog, λ)
            append!(gaplog, abs(dot(∇f(x, α), real.(q - x))) / abs(f(x)))
            append!(flog, f(x))
        end


        if (printOption == "full")
            println(string(t) * ": ", abs(dot(∇f(x, α), real.(q - x))) / abs(f(x)), " ", f(x))
        elseif (printOption == "partial")
            if (t % 10 == 1)
                print(".")
            end
        end
        t = t + 1
    end
    # store the result of the last iteration
    # only if the last iteration is not a power of 2
    # otherwise duplicate
    if !ispow2(t-1) 
        append!(tlog, t-1)
        push!(xlog, x)
        push!(vlog, v)
        push!(qlog, q)
        append!(λlog, λ)
        append!(gaplog, abs(dot(∇f(x, α), real.(q - x))) / abs(f(x)))
        append!(flog, f(x))
    end

    if (printOption == "partial")
        println()
    end
    return (z=z, x=x, t=t, tlog=tlog, xlog=xlog, qlog=qlog, vlog=vlog, λlog=λlog, gaplog=gaplog, flog=flog)
end