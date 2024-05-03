graphs = readlines(homedir()*"/datasets/graphs/all_moderate.txt")
tol = 1e-2
seed = 1

open(homedir()*"/SCAMSv2-new/SCAMS/test_maxcut.txt", "w") do io
    for graph in graphs
        println(io, "ulimit -m $((16 * 1024 * 1024)); /p/mnt/software/julia-1.7.1/bin/julia --project Test.jl --seed $seed --tol $tol --graph $graph")
    end
end