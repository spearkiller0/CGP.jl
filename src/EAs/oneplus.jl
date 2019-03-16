export oneplus

function oneplus(nin::Int64, nout::Int64, fitness::Function;
                 ctype::DataType=CGPChromo, seed::Int64=0, expert::Any=nothing,
                 id::String="")
    population = Array{ctype}(Config.lambda)
    for i in eachindex(population)
        population[i] = ctype(nin, nout)
    end
    if expert != nothing
        population[1] = expert
    end
    best = population[1]
    max_fit = -Inf
    eval_count = 0
    fits = -Inf*ones(Float64, Config.lambda)

    while eval_count < Config.total_evals
        log_gen = false
        for p in eachindex(population)
           
            if fits[p] == -Inf
                println("Evaluation:",eval_count+1," - population:",p)
                fit = fitness(population[p])
                println("Accuracy: ",fit)
                eval_count += 1
                if fit >= max_fit
                    best = clone(population[p])
                    if fit > max_fit
                        max_fit = fit
                        log_gen = true
                    end
                end
                fits[p] = fit
                if eval_count == Config.total_evals
                    log_gen = true
                    break
                end
            end
        end
        
        eval(Config.log_function)(id, seed, eval_count, max_fit, best, GA,
                                  ctype, log_gen)

        if eval_count == Config.total_evals
            break
        end

       
        
        # selection
        fits .= -Inf
        for p in eachindex(population)
            population[p] = mutate(best)
        end
        

        # size limit
        for i in eachindex(population)
            if length(population[i].nodes) > Config.node_size_cap
                population[i] = ctype(nin, nout)
                fits[i] = -Inf
            end
        end
        println("_________________________________________\n")
    end
    println("_________________________________________")
    println("Max Accuracy:", max_fit)
    println("_________________________________________\n")
    max_fit, best.genes
end
