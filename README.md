# SCAMSv2
Scalable Conditional gradient Algorithm for Max-Cut SDP.
This repo contains code for the paper 
```
A Scalable Frank-Wolfe-Based Algorithm for the Max-Cut SDP
```
and some modification for batch testing it.

# Installation
You need to install `Julia 1.7.1` and then you can enter
the folder `SCAMS` and start Julia via `julia --project`,
which reads the Mainfest.toml and Project.toml files to
install the required packages. To confirm you are using
the correct environment, type `]` in Julia REPL and make sure
the environment is `(SCAMS)`. 

# Data
The only input data required for the solver is an adjacency 
matrix for a graph. Here we provide some example data from
[Gset](https://www.cise.ufl.edu/research/sparse/matrices/Gset/)
and we preprossed them to flip all negative edges as required.

The data loading is implemented in the `read_graph` function
under `Test.jl`, you can customize it based on the data format
you have.  

# Running
You can call `test(graph_name, tol)` in Julia REPL, or you can
use the batch test tool we provide. In `gen_test.jl`, modify 
the list of graphs you want to test SCAMS on and then run it
to generate the batch test txt `test_MaxCut.txt` (we include 
one example here). After you run `gen_test.jl` and the batch test txt file is generated, 
you can parallel the benchmarking via using GNU parallel, e.g. 
```
cat test_MaxCut.txt | parallel --jobs 9 --timeout 28800 {}
```
Here `--timeout` specifies the time limit, and `--jobs` means the number of 
parallel jobs you want to run. Keep in mind that you should not set `--jobs` 
larger than the number of cores you have, which will downgrade the performance.

You may need to install **ulimit** and **parallel** to enable 
memory usage limiting and parallel testing. 
Also, for fair benchmarking, we disable multithreading here by default. To make sure multithreading is off, 
you may also want to set the following environment variables.
```
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

# Contributions
The original solver codes were created by Chi Bach Pham under the supervision of Dr. James Saunderson and Dr. Wynita Griggs at Monash University, Australia.
Later improvements were made by Yufan Huang at Purdue University.
