include("./ArnoldiMethodMod.jl")
include("sampling.jl")
using .ArnoldiMethodMod
using LinearAlgebra, SparseArrays

function LMOtrueGrad(A, v)
    ∇P = spzeros(m, m)
    for i in 1:n
        a = A[:, i]
        ∇P = ∇P - a * a' / (2 * sqrt(v[i]))
        #∇P = ∇P - a*a'/(4*v[i]^(3/4))
    end
    (eig, eigv) = eigs(∇P)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, w)
    return w, q, eig[1]
end

function ArnoldiGrad(A, v)
    decomp, history = partialschur(A, v, tol=1e-6, which=LM())
    eig, eigv = partialeigen(decomp)
    #disp(eig)
    #disp(eigv)
    w = eigv[:, 1]
    w = w / norm(w)
    q = B(A, w)
    return w, q, eig[1]
end

A = readdlm("graphbackup.csv", ',', Float64, '\n')
global m = size(A, 1)
global n = size(A, 2)
A = A / 2
A_s = sparse(A)
z, v = genSample2(A)

#=
#sleep(15)
w, q, e = ArnoldiGrad(A, v)
display(w)
println()
display(q)
println()
display(e)
println()
#sleep(15)
w, q, e = LMOtrueGrad(A, v)
display(w)
println()
display(q)
println()
display(e)
println()
=#

#display(@benchmark ArnoldiGrad(A, v))
#display(@benchmark LMOtrueGrad(A, v))
#=
function solvesampTrue(A, v0, z0;)
    v = v0
    z = z0
    t = 0
    start = t0
    gamma = 2 / (t + start)
    w, q, λ = LMOtrueGrad(A, v)
    while abs(dot(q - v, ∇g(v))) / abs(f(A, v)) > 1e-4 && t < 1000
        t = t + 1
        z = sqrt(1 - gamma) * z + sqrt(gamma) * w * rand(Normal(0, 1))
        v = (1 - gamma) * v + gamma * q

        gamma = 2 / (t + start)
        w, q, λ = LMOtrueGrad(A, v)
    end
    x = randn(n)
    mul!(x, A', z)
    x = sign.(x)
    result = (val=f(A, v), x=x, v=v, z=z, t=t)
    return result
end

result = solvesampTrue(A, v, z, 2)

display(result.t)
w, q, e = ArnoldiGrad(A, result.v)
display(w)
println()
display(e)
println()
#sleep(15)
w, q, e = LMOtrueGrad(A, result.v)
display(w)
println()
display(e)
println()
=#
display(@benchmark ArnoldiGrad(A, v))
display(@benchmark LMOtrueGrad(A, v))