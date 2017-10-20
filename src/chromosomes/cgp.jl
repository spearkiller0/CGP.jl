export CGPChromo, node_genes, get_positions

# vanilla CGP

type CGPChromo <: Chromosome
    genes::Array{Float64}
    nodes::Array{CGPNode}
    outputs::Array{Int64}
    order::Array{Int64}
    nin::Int64
    nout::Int64
end

function CGPChromo(genes::Array{Float64}, nin::Int64, nout::Int64)::CGPChromo
    num_nodes = Int64(ceil((length(genes)-nin-nout)/4))
    rgenes = reshape(genes[(nin+nout+1):end], (4, num_nodes))'
    connections = Array{Int64}(2, nin+num_nodes)
    connections[:, 1:nin] = zeros(2, nin)
    for i in 1:num_nodes
        connections[:, nin+i] = Int64.(ceil.(rgenes[i, 1:2]*(nin+i-1)))
    end
    functions = Array{Function}(nin+num_nodes)
    functions[1:nin] = Config.f_input
    functions[(nin+1):end] = Config.functions[(
        Int64.(ceil.(rgenes[:, 3]*length(Config.functions))))]
    outputs = Int64.(ceil.(genes[nin+(1:nout)]*(nin+num_nodes)))
    active = find_active(nin, outputs, connections)
    params = [zeros(nin); 2.0*rgenes[:, 4]-1.0]
    nodes = Array{CGPNode}(nin+num_nodes)
    for i in 1:(nin+num_nodes)
        nodes[i] = CGPNode(connections[:, i], functions[i], active[i], params[i])
    end
    order = collect(1:length(nodes))
    CGPChromo(genes, nodes, outputs, order, nin, nout)
end

function CGPChromo(nin::Int64, nout::Int64)::CGPChromo
    CGPChromo(rand(nin+nout+4*Config.num_nodes), nin, nout)
end

function CGPChromo(c::CGPChromo)::CGPChromo
    mutate_genes(c)
end

function node_genes(c::CGPChromo)
    4
end

function get_positions(c::CGPChromo)
    collect(1:(length(c.nodes)))/length(c.nodes)
end
