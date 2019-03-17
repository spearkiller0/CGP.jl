using CGP
using Logging
using ArgParse
using JSON
using Images
using FileIO
using Base.Threads
using Dates
CGP.Config.init("cfg/image_caption.yaml")


function overprint(str)  
    print("\e[2K")
    print("\e[1G")
    print(str)
end

function computeParallel(c::Chromosome, startIndex::Int64, endIndex::Int64, data::Array{Any,2}, images::Dict{Any,Any}, nin::Int64, nout::Int64, accuracy::Base.Threads.Atomic{Float64})::Float64
    @threads for d in startIndex:endIndex

        imagePath = string("data/images/",data[d,1],".jpg")
        img = images[imagePath]
        r = Float64.(red.(img))
        g = Float64.(green.(img))
        b = Float64.(blue.(img))

        outputs = process(c,[r,g,b,data[d,2:nin-1]...])
        if indmax(outputs)==data[d,nin-1]
            atomic_add!(accuracy,1.0)
        end
    end
    accuracy[]
end

function blueScore(c::Chromosome, data::Array{Any,2}, nin::Int64, nout::Int64)
    
    accuracy =  0.0
    batchSize = 1000
    nsamples = size(data, 1)
    nbatches = Int64(ceil(nsamples/batchSize))
    
    
    for batch in 1:nbatches

        startIndex = (batch-1)*batchSize + 1 
        endIndex = batch*batchSize<nsamples? batch*batchSize:nsamples
        
        images = Dict()
        prevImgPath = string("data/images/",data[startIndex,1],".jpg")
        img = load(prevImgPath)
        images[prevImgPath] = img


        
        println("batch:",batch,"/",nbatches)
        println("loading images batch: $(startIndex):$(endIndex)")
        for sample in startIndex:endIndex
            imgPath= string("data/images/$(data[sample,1]).jpg")
            if imgPath != prevImgPath
                img = load(imgPath)
                prevImgPath=imgPath
                images[imgPath] = img
            end
        end
        println("image Batch Loaded Successfully")
        println("<<< Training >>>")

        now = Dates.now()
        accuracy += computeParallel(c, startIndex, endIndex, data[startIndex:endIndex,:], images, nin, nout, Threads.Atomic{Float64}(0.0))
        println("paralle time: $(Dates.value(Dates.now()-now)/1000)s")

    end

    accuracy /= nsamples
    println()
    accuracy
end

function predict(c::Chromosome, path::String, nin::Int64)
    
        img = load(path)
        r = Float64.(red.(img))
        g = Float64.(green.(img))
        b = Float64.(blue.(img))
        sequence =  zeros(Float64, nin-3)
        sequence[nin-3]=1
        for i = 1:(nin-3)
            outputs = process(c,[r,g,b,sequence...])
            for j = 1:(nin-4)
                sequence[j]=sequence[j+1]
            end
            sequence[nin-3] = indmax(outputs)
            if indmax(outputs) == 7
                
                break
            end
        end
        println(sequence)

end

function MSE(c::Chromosome, data::Array{Float64}, nin::Int64, nout::Int64)
    accuracy = 0
    nsamples = size(data, 2)
    for d in 1:nsamples
        outputs = process(c, data[d, 2:nin+1])
        if indmax(outputs) == data[d,nin+1]
            accuracy += 1
        end
    end
    accuracy /= nsamples
    accuracy
end


function get_args()
    s = ArgParseSettings()

    @add_arg_table(
        s,
        "--id", arg_type = String, default = "image_caption", 
        "--seed", arg_type = Int, default = 0,
        "--log", arg_type = String, default = "image_caption.log",
        "--ea", arg_type = String, default = "oneplus",
        "--chromosome", arg_type = String, default = "CGPChromo",
        "--data", arg_type = String, required = true,
        "--meta", arg_type = String, required = true,
        "--fitness", arg_type = String, default = "blueScore",
    )

    parse_args(CGP.Config.add_arg_settings!(s))
end


function read_data_descriptor(dMeta::String, dData::String)
    j = JSON.parsefile(dMeta)
    seqLen = j["meta"]["Sequence Length"]
    outputSize = j["meta"]["Vocab Size"]
    wordDict = j["dict"]

    data = readdlm(dData, ' ')
    data[:,2:seqLen+1] = Float64.(data[:,2:seqLen+1])
    #seqLen + r+g+b
    return seqLen + 3, outputSize, data, data, wordDict
end

if ~isinteractive()
    args = get_args()
    CGP.Config.init(Dict([k=>args[k] for k in setdiff(
        keys(args), ["seed", "log", "ea", "chromosome", "data", "meta", "fitness","id"])]...))

    srand(args["seed"])
    Logging.configure(filename=args["log"], level=INFO)
    
    println("Loading Dataset ...")
    nin, nout, train, test, word_indx = read_data_descriptor(args["meta"],args["data"])
    ea = eval(parse(args["ea"]))
    ctype = eval(parse(args["chromosome"]))
    fitness = eval(parse(args["fitness"]))
    trainFit = x->fitness(x, train, nin, nout)

    println("Training ...")
    maxfit, best = ea(nin, nout, trainFit, seed=args["seed"], id=args["id"], ctype=ctype)
    best_ind = ctype(best, nin, nout)
    test_fit = fitness(best_ind, test, nin, nout)
    
    Logging.info(@sprintf("T: %d %0.8f %0.8f %d %d %s %s",
                        args["seed"], maxfit, test_fit,
                        sum([n.active for n in best_ind.nodes]),
                        length(best_ind.nodes), args["ea"], args["chromosome"]))
    Logging.info(@sprintf("E%0.6f", -maxfit))

    predict(best_ind,"data/images/2a.jpg",nin)
    #render_genes(best, args; ctype=ctype)
end
