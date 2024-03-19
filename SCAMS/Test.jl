include("SCAMS.jl")
using Plots
using DelimitedFiles
using SparseArrays

function GenGraph(N, M)
    C = spzeros((N, N))
    for i = 1:N-1
        v = rand(1:N)
        while i == v || C[i, v] == -1
            v = rand(1:N)
        end
        C[i, v] = -1
        C[v, i] = -1
        C[i, i] += 1
        C[v, v] += 1
    end
    for i = N:M
        u = rand(1:N)
        v = rand(1:N)
        while u == v || C[u, v] == -1
            v = rand(1:N)
        end
        C[u, v] = -1
        C[v, u] = -1
        C[u, u] += 1
        C[v, v] += 1
    end
    return C
end

function run(C, N, M; ε0=-2.0)
    C = C / 4
    x = Array(diag(C) / M)
    z = rand(Normal(0, 1), (N, 1)) .* sqrt.(x)
    # eigSolver option A: Arnoldi method(recommended); P: Power Iteration; F: Full eigenvalue
    # printOption option "full", "partial", "none"
    ans = SCAMS(C, x, z, ε=10^(ε0), printOption="none", eigSolver="A")
    # ans tuple 
    # ans.z: sample; ans.x: iterate; ans.t: max Iteration
    # ans.xlog, ans.qlog, ans.vlog, ans.λlog: history of LMO routine outputs
    # ans.gaplog: Frank-Wolfe gap history; ans.flog: f(x) history

    println(sign.(ans.z))
    println("\n\nf(x):      ", ans.flog[1], "->", ans.flog[end])
    println("RFWgap(x): ", ans.gaplog[1], "->", ans.gaplog[end])
    println("Solved in ", ans.t, " iteration\n")
    plot!(log10.(1:length(ans.gaplog)), log10.(ans.gaplog), aspect_ratio=:equal)
    savefig("plot.png")
end

N = 5
M = 8
C = GenGraph(N, M)
display(C)
run(C, N, M)