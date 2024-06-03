graphs = ["G$i" for i = 1:9] # graphs you want to test on 
tol = 1e-2
seed = 0

open(homedir()*"/SCAMSv2/SCAMS/test_MaxCut.txt", "w") do io
    for graph in graphs
        println(io, "ulimit -d $((16 * 1024 * 1024));"*
        "cd ~/SCAMSv2/SCAMS;"*
        "/p/mnt/software/julia-1.7.1/bin/julia --project Test.jl --seed $seed --tol $tol --graph $graph")
    end
end