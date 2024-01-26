using JuMP, DiffOpt, LinearAlgebra, MosekTools, COSMO, ProgressMeter, Random, Statistics, FileIO, Dates, Comonicon, Sobol
const CHOSEN_OPTIM = Mosek.Optimizer
# you can switch this to  COSMO.Optimizer if you don't have Mosek license
# but you should consider getting a Mosek license if you can

include("lp_and_ama.jl")
const TEST_SAMPLES=20000
function namedtuple_to_csv_line_str(nt)
    # print all entries of namedtuple as csv
    str = ""
    for (k, v) in pairs(nt)
        str *= string(v) * "_"
    end
    str
end


function gridsearch(num_agents, num_items, num_samples, seed, mdptype::Type{T}, dist_type; optimize_weights=false,num_samples_evaluation=1000,γ=1.0) where {T<:MDP}
    @show seed
    Random.seed!(seed)
    #DANGER γ needs to be set from now on as a variable! set default to 1.0
    mdp = mdptype(num_agents, num_items, γ, dist_type)
    @show length(mdp.state_list)
    lp = MDPLinearProgram(mdp)
    # initialization matters here, even for this "trivial" problem
    # boosts = [0.3 0.1 0.2]
    # boosts = rand(size(lp.x, 1), size(lp.x, 2))
    # boosts = zeros(size(lp.x, 1), size(lp.x, 2))

    if mdptype==AuctionMDP 
        bounds = (- mdp.n_agents, mdp.n_agents)
    elseif mdptype==ScheduleMDP
        bounds = (- mdp.n_agents * max(mdp.n_agents,3), mdp.n_agents * max(mdp.n_agents,3))
    elseif mdptype == GridMDP && dist_type == "onecell"
        bounds = (- mdp.n_agents/(1-mdp.γ), mdp.n_agents/(1-mdp.γ))
        #bounds = (-1,1)
    elseif mdptype == GridMDP && dist_type == "uniform"
        bounds = (- mdp.n_agents*mdp.n_items^2/(1-mdp.γ), mdp.n_agents*mdp.n_items^2/(1-mdp.γ))
    else
        #throw not defined error
        error("MDP type not defined")
    end

    best_value = -Inf
    best_point = nothing

    weights = ones(mdp.n_agents)

    vcg_weights = ones(lp.mdp.n_agents)
    vcg_ama_params = AMAParams(vcg_weights, zeros(size(lp.x, 1), size(lp.x, 2)))


    start_time = Dates.now()
    best_performance = -Inf
    best_params = nothing
    ama_std=0
    # Create a Sobol sequence for weights
    s_weights = SobolSeq(mdp.n_agents)

    # Create a Sobol sequence for boosts
    s_boosts = SobolSeq(size(lp.x, 1) * size(lp.x, 2))

    for i in 1:num_samples
        if optimize_weights
            weights = Sobol.next!(s_weights)
            weights = weights ./ sum(weights)
        end
        current_boosts_raw = Sobol.next!(s_boosts)
        current_boosts = bounds[1] .+ (bounds[2] - bounds[1]) .* reshape(current_boosts_raw, size(lp.x, 1), size(lp.x, 2))
       
        
        ama_params = AMAParams(weights, current_boosts)
        #TODO make num_samples a second parameter
        performance, current_ama_std = expectedperformance(lp, ama_params, num_samples=num_samples_evaluation, α=0.00, require_optimal=true)
    
        if performance > best_performance
            best_performance = performance
            best_params = ama_params
            ama_std = current_ama_std
        end

        if i % 100 == 0 #So we can see roughly how far the cluster gets
            println("Iteration $i, best_performance: $best_performance")
            println("Iteration $i, best_params: $best_params")
        end

    end



    end_time = Dates.now()
    
    best_performance, ama_std = expectedperformance(lp, best_params, num_samples=(num_samples_evaluation*5), α=0.00, require_optimal=true) # VCG revenue
    vcg_performance, vcg_std = expectedperformance(lp, vcg_ama_params, num_samples=(num_samples_evaluation*5), α=0.00, require_optimal=true) # VCG revenue
    vcg_revenue, vcg_std_rev = expectedrevenue(lp, vcg_ama_params, num_samples=(num_samples_evaluation*5), α=0.00, require_optimal=true) # VCG revenue
    ama_revenue, ama_std_rev = expectedrevenue(lp, best_params, num_samples=(num_samples_evaluation*5), α=0.00, require_optimal=true) # VCG revenue

    @show best_performance
    @show ama_std
    @show vcg_performance
    @show vcg_std
    @show ama_revenue
    @show ama_std_rev
    @show vcg_revenue
    @show vcg_std_rev
    @show end_time - start_time
    @show best_params
    # print csv line consisting of seed, vcg_revenue, optimized_revenue
    println("*****************")
    (
        (method="grid",mdptype=mdptype, dist_type= dist_type,num_agents=num_agents, num_items=num_items, seed=seed, num_samples=num_samples,vcg_performance=vcg_performance, vcg_std=vcg_std,
            vcg_revenue=vcg_revenue, vcg_std_rev=vcg_std_rev, ama_revenue=ama_revenue,ama_std_rev=ama_std_rev, ama_performance=best_performance, ama_std=ama_std, runtime=(end_time - start_time),num_samples_evaluation=num_samples_evaluation),
        (ama=best_params, vals=best_performance)
    )
end



@main function main(num_agents::Int64 =2, num_items::Int64=3, num_samples::Int64=10, start_seed::Int64=42, mdptype_str::String="auction", dist_type::String="uniform", num_samples_evaluation::Int64=1000,γ::Float64=1.0)
    # @show ARGS
    # num_agents = parse(Int64, ARGS[1])
    # num_items = parse(Int64, ARGS[2])
    # num_samples = parse(Int64, ARGS[3])
    # num_training_iters = parse(Int64, ARGS[4])
    # start_seed = parse(Int64, ARGS[5])
    # num_trials = parse(Int64, ARGS[6])
    # if length(ARGS) > 6
    #     mdptype_str = (ARGS[7])
    # else
    #     mdptype_str = "auction"
    # end
    # if length(ARGS) > 7
    #     lr = parse(Float64, ARGS[8])
    # else
    #     lr = 0.01
    # end

    # if length(ARGS) > 8
    #     dist_type = ARGS[9]
    # else
    #     dist_type = "uniform"
    # end
    @show num_agents
    @show num_items
    @show num_samples
    @show start_seed
    @show mdptype_str
    @show dist_type

    all_results = []

    
    if mdptype_str == "auction"
        mdptype = AuctionMDP
    elseif mdptype_str == "schedule"
        mdptype = ScheduleMDP
    elseif mdptype_str == "matching"
        mdptype = MatchingMDP
    elseif mdptype_str == "grid"
        mdptype = GridMDP
    else
        error("invalid mdptype")
    end

    filedir = "results/"
    # create results if it doesn't exist
    if !isdir(filedir)
        mkdir(filedir)
    end
    csv_filename = "results_" * mdptype_str * "_" * dist_type * "_" * string(num_agents) * "_" * string(num_items) * "_" * string(num_samples) * "_" * string(start_seed) * "_" *  "_reg"  * ".csv"

    @show mdptype
    #start_seed += 1
    boostvals_filename = "boostvals_" * mdptype_str * dist_type * "_" * string(num_agents) * "_" * string(num_items) * "_" * string(num_samples) * "_" * string(start_seed) * "_" *  "_reg"  * ".jld2"
    use_weights = dist_type == "asymmetric"
    results, boosts_and_vals = gridsearch(num_agents, num_items, num_samples, start_seed, mdptype, dist_type, optimize_weights=use_weights, num_samples_evaluation=num_samples_evaluation,γ=γ)
    # save boosts and vals
    save(filedir * boostvals_filename, "boosts_and_vals", boosts_and_vals)
    push!(all_results, results)


    for result in all_results
        # print all entries of namedtuple as csv
        println(namedtuple_to_csv_line_str(result))

    end
    open(filedir * csv_filename, "w") do io
        write(io, join(keys(all_results[1]), "\t") * "\n")
        for result in all_results
            # print all entries of namedtuple as csv
            write(io, namedtuple_to_csv_line_str(result) * "\n")
        end
    end

end