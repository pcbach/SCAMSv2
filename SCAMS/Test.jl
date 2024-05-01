include("SCAMS.jl")
using Plots
using DelimitedFiles
using SparseArrays
using JSON
using Random

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
    for i = N+1:M
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

function solve(C, N, M; ε=0.01)
    C = C / 4
    x = Array(diag(C) / M)
    z = rand(Normal(0, 1), (N, 1)) .* sqrt.(x)
    # eigSolver option A: Arnoldi method(recommended); P: Power Iteration; F: Full eigenvalue
    # printOption option "full", "partial", "none"
    # here we divide the ε by 2 because 

    # SCAMS gives one (1 ± ε)-approximation of sqrt root of the maxcut value 
    # which in turn gives one (1 ± 2ε)-approximation of the maxcut value
    # so to achieve desired accuracy, we need to half the ε 
    ans = SCAMS(C, x, z, ε=ε/2, printOption="none", eigSolver="A")
    # ans tuple 
    # ans.z: sample; ans.x: iterate; ans.t: max Iteration
    # ans.xlog, ans.qlog, ans.vlog, ans.λlog: history of LMO routine outputs
    # ans.gaplog: Frank-Wolfe gap history; ans.flog: f(x) history

    println("\n\nf(x):      ", ans.flog[1], "->", ans.flog[end])
    println("RFWgap(x): ", ans.gaplog[1], "->", ans.gaplog[end])
    println("Solved in ", ans.t, " iteration\n")
    #plot!(log10.(1:length(ans.gaplog)), log10.(ans.gaplog), aspect_ratio=:equal)
    #savefig("SCAMS_plot.png")
    return ans
end

using MAT

function read_graph(
    filename::String;
    filefolder::String=homedir()*"/datasets/graphs/",
)
    filepath = filefolder*filename*".mat"
    data = matread(filepath) 
    return data 
end

BLAS.set_num_threads(1)

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "--graph"
        arg_type = String
        default = "G1" 
        help = "Name of the graph"
    "--tol"
        arg_type = Float64
        default = 1e-2
        help = "Tolerance for relative Frank-Wolfe duality gap"
    "--seed"
        arg_type = Int64
        default = 0
        help = "Random seed"
end

# parse the command line arguments
args = parse_args(s)

# fix the random seed for reproducibility
seed = args["seed"]
Random.seed!(args["seed"])

tol = args["tol"]
dataset = args["graph"]

function test(dataset, tol) 
    @info "Solving MaxCut for $dataset"
    A = read_graph(dataset)["A_abs"]
    d = sum(A, dims=1)[1, :]
    C = Diagonal(d) - A
    N = size(C, 1)
    M = div(nnz(A), 2) 
    res = @timed ans = solve(C, N, M; ε=tol)
    ans = Dict(pairs(ans))
    ans[:time] = res.time

    short_ans = Dict(
        #:z => ans[:z],    
        #:x => ans[:x],
        :t => ans[:t],
        #:λlog => ans[:λlog],
        :flog => ans[:flog],
        :gaplog => ans[:gaplog],
        :time => ans[:time]
    )
    @show res.time
    return ans, short_ans
end

# julia requires warmup
test("G1", tol)

# run the test
ans, short_ans = test(dataset, tol)

output_path=homedir()*"/SDPLR.jl/output/MaxCut/"*dataset*"/SCAMS/"

mkpath(output_path)
open(output_path*"SCAMS-tol-$tol-seed-$seed.json", "w") do f
    JSON.print(f, ans, 4)
end
open(output_path*"SCAMS-short-tol-$tol-seed-$seed.json", "w") do f
    JSON.print(f, short_ans, 4)
end