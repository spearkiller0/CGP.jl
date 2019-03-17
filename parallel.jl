using Base.Threads
using FileIO

function test1(x::Array{Float64,1},y::Array{Float64,1})::Array{Float64,1}
    println("number of threads:",Threads.nthreads())
    z = zeros(1000000)
    @threads for i in 1:5
        load("667626_18933d713e.jpg")
    end
    z
end

function test2(x::Array{Float64,1},y::Array{Float64,1})::Array{Float64,1}
    z = zeros(1000000)
    for i in 1:1000000
        z[i] = x[i]+y[i]
    end
    z
end
x = rand(111711111)
y = rand(111711111)
@time test1(x,y)

@time test2(x,y)