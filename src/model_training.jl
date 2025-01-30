using Flux
using BSON
using Zygote
using Graphs
using MLUtils
using DataFrames
using Statistics
import Optimisers
using Transformers
using ClusterManagers
using GraphNeuralNetworks
using SimpleWeightedGraphs
using Transformers.Layers
using Distributed
using BSON: @save

addprocs(SlurmManager(8), A="trans-net", t="32:00:00", c="7")

@everywhere using Flux, BSON, Zygote, Graphs, MLUtils, DataFrames
@everywhere using Statistics, Transformers, ClusterManagers
@everywhere using GraphNeuralNetworks ,SimpleWeightedGraphs
@everywhere using Transformers.Layers
@everywhere import Optimisers
@everywhere using BSON: @save


@everywhere begin
    applicants = BSON.load("applicants.bson")
    applicants = applicants[:applicants]
end

@everywhere begin
    ReducedApplicantsEmbeddings = BSON.load("ReducedApplicantsEmbeddings.bson")
    ReducedApplicantsEmbeddings = ReducedApplicantsEmbeddings[:ReducedApplicantsEmbeddings]
end

@everywhere begin
    jobAdvertEmbeddings = BSON.load("jobAdvertEmbeddings.bson")
    jobAdvertEmbeddings = jobAdvertEmbeddings[:jobAdvertEmbeddings]
end

@everywhere begin
    occurrences = Dict{String, Int}()
    for element in applicants.Job_Id
        occurrences[element] = get(occurrences, element, 0) + 1
    end
    sortedlements = sort(occurrences)
end

@everywhere function build_graphs(sortedlements, applicants, ReducedApplicantsEmbeddings, vector_embeddings)
    startIndex = 1
    start = 1
    HeteroGraph =GNNHeteroGraph[]

    for (key, value) in sortedlements
            max_value = maximum(values(sortedlements)) + 1
            end_value = start + value - 1

            num_nodes  = Dict(:JobAdvert =>1, :Applicants =>max_value )
            data = ((:JobAdvert,:applied, :Applicants) => (fill(1, max_value), collect(1:max_value)))

            JobAdvertFeatures = jobAdvertEmbeddings[start:end_value, :][1:1,:]

            ApplicantFeatures = vector_embeddings[start:end_value, :]'

            ApplicantFeatures = size(ApplicantFeatures, 2) < max_value ? hcat(ApplicantFeatures, zeros(eltype(ApplicantFeatures), size(ApplicantFeatures, 1), max_value - size(ApplicantFeatures, 2))) : ApplicantFeatures

            EdgeFeatures = applicants.Interview_Score[start:end_value]
            EdgeFeatures = vcat(EdgeFeatures, zeros(Int, max(0, max_value - length(EdgeFeatures))))

            ndata = Dict(:JobAdvert => vec(JobAdvertFeatures), :Applicants => ApplicantFeatures);

            edata =((:JobAdvert,:applied, :Applicants) => EdgeFeatures)


            push!(HeteroGraph,GNNHeteroGraph(data;num_nodes,ndata,edata))

            start = end_value + 1
    end
    return HeteroGraph
end

@everywhere begin
    all_graphs = build_graphs(sortedlements, applicants, ReducedApplicantsEmbeddings, jobAdvertEmbeddings)

    train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.8, shuffle=false)

    train_loader = DataLoader(train_graphs, batchsize=1, shuffle=true, collate=true)
end

test_loader = DataLoader(test_graphs, batchsize=1, shuffle=true, collate=true)

@everywhere begin
    N = 6
    hidden_dim = 337
    head_num = 8
    head_dim = 64
    ffn_dim = 2048
    output_sequence_length = maximum(values(sortedlements))
    nin, nhidden, nout = 300, 64, 1

    # define 6 layer of transformer
    encoder_trf = Transformer(TransformerBlock, N, head_num, hidden_dim, head_dim, ffn_dim);

    # define 6 layer of transformer decoder
    decoder_trf = Transformer(TransformerDecoderBlock, N, head_num, hidden_dim, head_dim, ffn_dim);

    const seq2seq = Seq2Seq(encoder_trf, decoder_trf);

    const trf_model = Layers.Chain(
        Layers.Parallel{(:encoder_input, :decoder_input)}( Dropout(0.1)),
        seq2seq,
    )

    function preprocess(src,trg)
        return (
        encoder_input = (hidden_state = src,),

        decoder_input = (hidden_state = trg,))
    end

    struct TransNetGNN
        GNN
        Interface_layer
        Transformer
    end

    function TransNetGNN(nin::Int, nhidden::Int, nout::Int)
        TransNetGNN(HeteroGraphConv((:JobAdvert, :applied, :Applicants) =>
                        ResGatedGraphConv(nin => nhidden, relu); aggr = +),
                        Chain(Dense(nhidden => 16),Dense(16 => 8),Dense(8 => nout)),
                        trf_model
            )
    end

    function (model::TransNetGNN)(g::GNNHeteroGraph, x)
    x = model.GNN(g, x)
    x = preprocess(vec(model.Interface_layer(x[:Applicants])),  g.edata[(:JobAdvert, :applied, :Applicants)].e)
            x = model.Transformer(x)
    return x
    end

    model = TransNetGNN(nin, nhidden, nout)

    const opt_rule = Optimisers.Adam(1e-4)
    const opt = Optimisers.setup(opt_rule, trf_model)

end

@everywhere function train_model(train_loader, model, opt)
    for i = 1:1200
        for g in train_loader
            decode_loss, (grad,) = Zygote.withgradient(model) do model
                ŷ = model(g, (JobAdvert = g.ndata[:JobAdvert].x, Applicants = g.ndata[:Applicants].x))
                Flux.mse(vec(ŷ.hidden_state[:,:]), g.edata[(:JobAdvert,:applied, :Applicants)].e)
            end

            i % 8 == 0 && @show decode_loss

            Optimisers.update!(opt, model, grad)
        end
    end
end

@sync @distributed for worker in workers()
    train_model(train_loader, model, opt)
end


function prediction()
        trg = Array{Float32}[]
        src = Array{Float32}[]
        for g in test_loader
                gnnOutput = model(g, (JobAdvert = g.ndata[:JobAdvert].x, Applicants = g.ndata[:Applicants].x))

                push!(trg,vec(vec(gnnOutput.hidden_state[:,:])))
                push!(src, g.edata[(:JobAdvert,:applied, :Applicants)].e)
        end
        return (y=vcat(src...),ŷ=vcat(trg...))
end


output = prediction()

@show Flux.mse(output[1], output[2])

@save "/home/mthomas/TransNetGNN/output.bson" output
