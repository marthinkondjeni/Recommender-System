using MLJ
using CSV
using CUDA
using Glob
using Flux
using Plots
using PDFIO
using Graphs
using MLUtils
using Markdown
using GraphPlot
using DataFrames
using BSON: @save
using Embeddings
using TextAnalysis
using Transformers
using Transformers.Layers
using SimpleWeightedGraphs
using GraphNeuralNetworks
using Flux: onehotbatch,onehot
using MLJMultivariateStatsInterface

function load_and_prepare_data()
    applicants = DataFrame(CSV.File("applicants.csv"))
    sort!(applicants, :Job_Id)
    return applicants
end

# Load applicants data
applicants = load_and_prepare_data()

@save "/home/mthomas/Others/applicants.bson" applicants

function process_job_adverts()
    files = Glob.glob("job_adverts/*.txt")
    extracted = Dict()
    for i in 1:length(files)
        open(files[i], "r") do file
            content = read(file, String)
            content = replace(content, '\n' => "")
            content = replace(content, '\uad' => ' ')
            content = replace(content, '®' => ' ')
            content = strip(content)
            extracted[splitext(basename(files[i]))[1]] = content
        end
    end
    return extracted
end

job_adverts_results = process_job_adverts()

JobAverts = DataFrame(Job_Id=collect(keys(job_adverts_results)), Job_Advert=collect(values(job_adverts_results)))

combined_data = leftjoin(applicants, JobAverts, on=:Job_Id)

function onehot_encode_column(column)
    return onehotbatch(unique(column), column)
end

encoded_columns = [onehot_encode_column(applicants[!, col]) for col in [
    :Job_Id, :Creation_Date, :Job_Title, :Business_Unit, :Country, :City_or_Town,
    :Institution, :Qualifications_Type, :Qualification_Name, :Other_Qualifications,
    :Professional_Qualification, :General_Work_Experience, :General_Availability,
    :Current_Employer, :Current_Title, :Job_Experience_Level, :Type_of_Employment,
    :Eligibility_Namibia, :Academic_Average, :Campus_Hire_or_Experienced_Hire, :Tax_Company_Experience
]]

categoricalEncoded = hcat(encoded_columns...)

embedding_dim = 5139
embedding_layer = Flux.Embedding(size(categoricalEncoded)[1], embedding_dim)

applicantsEmbeddings = embedding_layer(categoricalEncoded)

@save "/home/mthomas/Others/applicantsEmbeddings.bson" applicantsEmbeddings

# Text processing using Corpus
function prepare_corpus(texts)
    docs = [StringDocument(text) for text in texts]
    crps = Corpus(docs)
    try
        prepare!(crps, strip_html_tags | strip_stopwords | strip_case | strip_pronouns | strip_prepositions | strip_indefinite_articles | strip_definite_articles |strip_patterns | strip_corrupt_utf8)
    catch
        # ignore preparation errors
    end
    stem!(crps)
    update_lexicon!(crps)
    return crps
end

crps = prepare_corpus(combined_data.Job_Advert)

embeddings = load_embeddings(Word2Vec)

function generate_emb(doc::AbstractDocument)
    tk = tokens(doc)
    emb_ind = [findfirst(y -> y == x, embeddings.vocab) for x in tk]
    filter!(x->x != nothing, emb_ind)
    sum(embeddings.embeddings[:, emb_ind], dims=2)
end

vector_embeddings = [generate_emb(doc) for doc in crps]

jobAdvertEmbeddings = hcat(vector_embeddings...)'

# Reduce dimensionality of applicant embeddings
model = PCA(maxoutdim=300)
mach = machine(model, applicantsEmbeddings, scitype_check_level=0) |> MLJ.fit!
applicantsProj = MLJ.transform(mach, applicantsEmbeddings)
ReducedApplicantsEmbeddings = hcat(applicantsProj...)

@save "/home/mthomas/Others/ReducedApplicantsEmbeddings.bson" ReducedApplicantsEmbeddings
