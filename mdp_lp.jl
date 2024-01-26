using JuMP, DiffOpt, LinearAlgebra, MosekTools, COSMO, ProgressMeter, Random, Statistics, FileIO, Dates, Comonicon
const CHOSEN_OPTIM = Mosek.Optimizer
# you can switch this to  COSMO.Optimizer if you don't have Mosek license
# but you should consider getting a Mosek license if you can


#duplicate
include("lp_and_ama.jl")
const TEST_SAMPLES=10000


# for testing purposes
function evalwithoutreg!(auctionlp::UnregMDP{T}, types, ama) where {T<:MDP}
    boosts = ama.boosts

    rew = @expression(auctionlp.model, 0.0 * auctionlp.x[1, 1])
    for state_ind in 1:length(auctionlp.mdp.state_list)
        for action_ind in 1:length(auctionlp.mdp.action_list)
            add_to_expression!(rew, (
                    sum(ama.weights .* reward_from_alloc(auctionlp.mdp, auctionlp.mdp.state_list[state_ind], auctionlp.mdp.action_list[action_ind], types)) +
                    boosts[state_ind, action_ind]),
                auctionlp.x[state_ind, action_ind])
        end
    end
    @objective(auctionlp.model, Max, rew)
    optimize!(auctionlp.model)
    @assert JuMP.termination_status(auctionlp.model) == MOI.OPTIMAL
end


function makespangrad(lp::MDPLinearProgram, types, ama, α)
    evalauction!(lp, types, ama, α)

    termination_status = JuMP.termination_status(lp.model)
    if termination_status != MOI.OPTIMAL
        println("WARNING: NOT OPTIMAL")
        return zeros(size(lp.x))
    end
    main_x = value.(lp.x)

    deriv_makespan_wrt_x = dmakespan_dx(lp.mdp, main_x, types)  # only works for ScheduleMDP

    MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], deriv_makespan_wrt_x[:])
    DiffOpt.reverse_differentiate!(lp.model)
    grad_obj = MOI.get(lp.model, DiffOpt.ReverseObjectiveFunction())

    JuMP.coefficient.(grad_obj, lp.x)
end

function makespangrad_wb(lp::MDPLinearProgram, types, ama, α)
    evalauction!(lp, types, ama, α)

    termination_status = JuMP.termination_status(lp.model)
    if termination_status != MOI.OPTIMAL
        println("WARNING: NOT OPTIMAL")
        return (zeros(lp.mdp.n_agents), zeros(size(lp.x)))
    end
    main_x = value.(lp.x)

    deriv_makespan_wrt_x = dmakespan_dx(lp.mdp, main_x, types)  # only works for ScheduleMDP

    MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], deriv_makespan_wrt_x[:])
    DiffOpt.reverse_differentiate!(lp.model)
    grad_obj = MOI.get(lp.model, DiffOpt.ReverseObjectiveFunction())

    b_grad = JuMP.coefficient.(grad_obj, lp.x)

    w_grads = zeros(lp.mdp.n_agents)
    for state = 1:length(lp.mdp.state_list)
        for action = 1:length(lp.mdp.action_list)
            rewards = reward_from_alloc(lp.mdp, lp.mdp.state_list[state], lp.mdp.action_list[action], types)
            w_grads .+= b_grad[state, action] .* rewards
        end
    end

    w_grads, b_grad
end


function revenuegradb_asw_envelope(lp::MDPLinearProgram, types, ama, α)
    # like revenue grad but with the envelope theorem for affine social welfare, ignoring regularization for those terms

    evalauction!(lp, types, ama, α)

    termination_status = JuMP.termination_status(lp.model)
    if termination_status != MOI.OPTIMAL
        println("WARNING: NOT OPTIMAL")
        return zeros(size(lp.x))
    end
    main_x = value.(lp.x)

    deriv_sw_wrt_x = dsw_dx(lp.mdp, main_x, types)

    MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], deriv_sw_wrt_x[:])
    DiffOpt.reverse_differentiate!(lp.model)
    grad_obj = MOI.get(lp.model, DiffOpt.ReverseObjectiveFunction())
    # each b(s,a) is just multiplied by a single x
    rev_grad_b = JuMP.coefficient.(grad_obj, lp.x)
    for i = 1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        evalauction!(lp, counterfactual_types, ama, α)
        counterfactual_x = value.(lp.x)
        rev_grad_b += (counterfactual_x - main_x) / ama.weights[i]
    end
    rev_grad_b
end

function revenuegrad_asw_envelope_wb(lp::MDPLinearProgram, types, ama, α)

    evalauction!(lp, types, ama, α)

    termination_status = JuMP.termination_status(lp.model)
    if termination_status == MOI.SLOW_PROGRESS
        println("WARNING: SLOW PROGRESS")
        return zeros(lp.mdp.n_agents), zeros(size(lp.x))
    end
    main_x = value.(lp.x)

    # calculate asw
    main_asw = asw(lp.mdp, main_x, types, ama)

    deriv_sw_wrt_x = dsw_dx(lp.mdp, main_x, types)

    MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], deriv_sw_wrt_x[:])
    DiffOpt.reverse_differentiate!(lp.model)
    grad_obj = MOI.get(lp.model, DiffOpt.ReverseObjectiveFunction())
    # each b(s,a) is just multiplied by a single x
    rev_grad_b = JuMP.coefficient.(grad_obj, lp.x)
    counterfactual_asws = zeros(lp.mdp.n_agents)
    for i = 1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        evalauction!(lp, counterfactual_types, ama, α)
        counterfactual_x = value.(lp.x)
        counterfactual_asws[i] = asw(lp.mdp, counterfactual_x, counterfactual_types, ama)
        rev_grad_b += (counterfactual_x - main_x) / ama.weights[i]
    end

    w_grads = zeros(lp.mdp.n_agents)
    for state = 1:length(lp.mdp.state_list)
        for action = 1:length(lp.mdp.action_list)
            rewards = reward_from_alloc(lp.mdp, lp.mdp.state_list[state], lp.mdp.action_list[action], types)
            # first term here is only correct under assumption that agent i's reward under policy -i is 0
            w_grads .+= ((rev_grad_b[state, action] .* rewards) ./ ama.weights)
        end
    end

    w_grads .+=  ( main_asw .* ( 1.0 ./ (ama.weights .^ 2)) )  .-  (counterfactual_asws ./ (ama.weights .^ 2)) 


    w_grads, rev_grad_b
end

function revenuegradb(lp::MDPLinearProgram, types, ama, α)
    evalauction!(lp, types, ama, α)

    termination_status = JuMP.termination_status(lp.model)
    if termination_status != MOI.OPTIMAL
        println("WARNING: NOT OPTIMAL")
        return zeros(size(lp.x))
    end
    main_x = value.(lp.x)

    deriv_sw_wrt_x = dsw_dx(lp.mdp, main_x, types)

    MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], deriv_sw_wrt_x[:])
    DiffOpt.reverse_differentiate!(lp.model)
    grad_obj = MOI.get(lp.model, DiffOpt.ReverseObjectiveFunction())
    # each b(s,a) is just multiplied by a single x
    rev_grad_b = JuMP.coefficient.(grad_obj, lp.x)

    main_dasw = dasw_dx(lp.mdp, main_x, types, ama)
    MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], main_dasw[:])
    # MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], ones(size(main_x))[:])
    DiffOpt.reverse_differentiate!(lp.model)
    asw_grad_obj = MOI.get(lp.model, DiffOpt.ReverseObjectiveFunction())
    asw_grad_b = JuMP.coefficient.(asw_grad_obj, lp.x) + main_x  ## add main_x too bc of chain rule
    # asw(x^*(b), b) depends on b in both arguments

    for i = 1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        evalauction!(lp, counterfactual_types, ama, α)
        counterfactual_x = value.(lp.x)
        deriv_asw_wrt_x = dasw_dx(lp.mdp, counterfactual_x, counterfactual_types, ama)
        MOI.set.(lp.model, DiffOpt.ReverseVariablePrimal(), lp.x[:], deriv_asw_wrt_x[:])
        DiffOpt.reverse_differentiate!(lp.model)
        counterfactual_grad_obj = MOI.get(lp.model, DiffOpt.ReverseObjectiveFunction())
        counterfactual_grad = JuMP.coefficient.(counterfactual_grad_obj, lp.x) + counterfactual_x
        rev_grad_b += (counterfactual_grad - asw_grad_b) / ama.weights[i]
    end
    rev_grad_b
end




# expectedperformance(lp::MDPLinearProgram{AuctionMDP}, ama; num_samples=1000, α=0.01, require_optimal=false) = expectedrevenue(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)
# expectedperformance(lp::MDPLinearProgram{MatchingMDP}, ama; num_samples=1000, α=0.01, require_optimal=false) = expectedrevenue(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)
# expectedperformance(lp::MDPLinearProgram{GridMDP},ama,num_samples=1000,α=0.01,require_optimal=false) = expectedrevenue(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)

function expectedrevenuegrad(lp::MDPLinearProgram, ama; num_samples=1000, α=0.01)
    # types = rand(lp.mdp.n_agents, num_samples)
    types = sampletypes(lp.mdp, num_samples)
    rev_grad_b = zeros(size(ama.boosts))
    for i = 1:num_samples
        rev_grad_b += revenuegradb_asw_envelope(lp, types[i], ama, α)
    end
    rev_grad_b / num_samples
end

function expectedrevenuegrad_wb(lp::MDPLinearProgram, ama; num_samples=1000, α=0.01)
    # types = rand(lp.mdp.n_agents, num_samples)
    types = sampletypes(lp.mdp, num_samples)
    rev_grad_b = zeros(size(ama.boosts))
    rev_grad_w = zeros(lp.mdp.n_agents)
    for i = 1:num_samples
        rev_grad_w_i, rev_grad_b_i = revenuegrad_asw_envelope_wb(lp, types[i], ama, α)
        rev_grad_b += rev_grad_b_i
        rev_grad_w += rev_grad_w_i
    end
    rev_grad_w / num_samples, rev_grad_b / num_samples
end

function expectedmakespangrad(lp::MDPLinearProgram, ama; num_samples=1000, α=0.01)
    # types = rand(lp.mdp.n_agents, num_samples)
    types = sampletypes(lp.mdp, num_samples)
    rev_grad_b = zeros(size(ama.boosts))
    for i = 1:num_samples
        rev_grad_b += makespangrad(lp, types[i], ama, α)
    end
    rev_grad_b / num_samples
end

function expectedmakespangrad_wb(lp::MDPLinearProgram, ama; num_samples=1000, α=0.01)
    # types = rand(lp.mdp.n_agents, num_samples)
    types = sampletypes(lp.mdp, num_samples)
    rev_grad_b = zeros(size(ama.boosts))
    rev_grad_w = zeros(lp.mdp.n_agents)
    for i = 1:num_samples
        grad_w, grad_b = makespangrad_wb(lp, types[i], ama, α)
        rev_grad_b += grad_b
        rev_grad_w += grad_w
    end
    rev_grad_w / num_samples, rev_grad_b / num_samples
end

function expectedmakespan(lp::MDPLinearProgram, ama; num_samples=1000, α=0.01, require_optimal=false)
    # types = rand(lp.mdp.n_agents, num_samples)
    types = sampletypes(lp.mdp, num_samples)
    all_revs = Array{Float64}(undef, num_samples)
    for i = 1:num_samples
        all_revs[i] = calcmakespan(lp, types[i], ama, α, require_optimal=require_optimal)
    end
    (sum(all_revs) / num_samples, std(all_revs))
end

#expectedperformance(lp::MDPLinearProgram{ScheduleMDP}, ama; num_samples=1000, α=0.01, require_optimal=false) = expectedmakespan(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)

at(x, i::CartesianIndex, val) = setindex!(copy(x), val, i)


function optimize_boosts!(lp::MDPLinearProgram{ScheduleMDP}, ama; num_samples=10, α=0.01, lr=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters
        vals[i], _ = expectedmakespan(lp, ama, num_samples=num_samples, α=α)
        grad = expectedmakespangrad(lp, ama, num_samples=num_samples, α=α)
        ama.boosts .-= lr * grad
    end
    (ama, vals)
end
function optimize_boosts!(lp::MDPLinearProgram{AuctionMDP}, ama; num_samples=10, α=0.01, lr=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
        grad = expectedrevenuegrad(lp, ama, num_samples=num_samples, α=α)
        ama.boosts .+= lr * grad
    end
    (ama, vals)
end

function optimize_boosts!(lp::MDPLinearProgram{GridMDP}, ama; num_samples=10, α=0.01, lr=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
        grad = expectedrevenuegrad(lp, ama, num_samples=num_samples, α=α)
        ama.boosts .+= lr * grad
    end
    (ama, vals)
end

function optimize_boosts!(lp::MDPLinearProgram{MatchingMDP}, ama; num_samples=10, α=0.01, lr=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
        grad = expectedrevenuegrad(lp, ama, num_samples=num_samples, α=α)
        ama.boosts .+= lr * grad
    end
    (ama, vals)
end

function optimize_weights_and_boosts!(lp::MDPLinearProgram{ScheduleMDP}, ama; num_samples=10, α=0.01, lr=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters
        vals[i], _ = expectedmakespan(lp, ama, num_samples=num_samples, α=α)
        w_grad, b_grad = expectedmakespangrad_wb(lp, ama, num_samples=num_samples, α=α)
        ama.boosts .-= lr * b_grad
        ama.weights .-= lr * w_grad
    end
    (ama, vals)
end

function optimize_weights_and_boosts!(lp::MDPLinearProgram{AuctionMDP}, ama; num_samples=10, α=0.01, lr=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
        w_grad, b_grad = expectedrevenuegrad_wb(lp, ama, num_samples=num_samples, α=α)
        ama.boosts .+= lr * b_grad
        ama.weights .+= lr * w_grad
    end
    (ama, vals)
end

function optimize_weights_and_boosts!(lp::MDPLinearProgram{MatchingMDP}, ama; num_samples=10, α=0.01, lr=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
        w_grad, b_grad = expectedrevenuegrad_wb(lp, ama, num_samples=num_samples, α=α)
        ama.boosts .+= lr * b_grad
        ama.weights .+= lr * w_grad
    end
    (ama, vals)
end


function runtrial(num_agents, num_items, num_samples, num_training_iters, seed, mdptype::Type{T}, lr, reg_strength, dist_type; optimize_weights=false, γ=1.0) where {T<:MDP}
    @show seed
    Random.seed!(seed)
    mdp = mdptype(num_agents, num_items, γ, dist_type)
    @show length(mdp.state_list)
    lp = MDPLinearProgram(mdp)
    # initialization matters here, even for this "trivial" problem
    # boosts = [0.3 0.1 0.2]
    boosts = rand(size(lp.x, 1), size(lp.x, 2))
    # boosts = zeros(size(lp.x, 1), size(lp.x, 2))
    vcg_weights = ones(lp.mdp.n_agents)
    ama_weights = ones(lp.mdp.n_agents)
    ama_params = AMAParams(ama_weights, boosts)
    vcg_ama_params = AMAParams(vcg_weights, zeros(size(lp.x, 1), size(lp.x, 2)))
    #vcg_revenue, vcg_std = expectedrevenue(lp, vcg_ama_params, num_samples=TEST_SAMPLES, α=0.00, require_optimal=true) # VCG revenue
    #vcg_performance, vcg_performance_std = expectedperformance(lp, vcg_ama_params, num_samples=TEST_SAMPLES, α=0.00, require_optimal=true) # VCG revenue
    #@show vcg_revenue
    #@show vcg_std

    start_time = Dates.now()
    if optimize_weights
        (ama_params, vals) = optimize_weights_and_boosts!(lp, ama_params, num_samples=num_samples, α=reg_strength, lr=lr, num_iters=num_training_iters)
    else
        (ama_params, vals) = optimize_boosts!(lp, ama_params, num_samples=num_samples, α=reg_strength, lr=lr, num_iters=num_training_iters)
    end
    end_time = Dates.now()
    vcg_revenue, vcg_std = expectedrevenue(lp, vcg_ama_params, num_samples=TEST_SAMPLES, α=0.00, require_optimal=true) # VCG revenue
    vcg_performance, vcg_performance_std = expectedperformance(lp, vcg_ama_params, num_samples=TEST_SAMPLES, α=0.00, require_optimal=true) # VCG revenue
    @show vcg_revenue
    @show vcg_std

    @show boosts
    ama_revenue, ama_std = expectedrevenue(lp, ama_params, num_samples=TEST_SAMPLES, α=0.0, require_optimal=true) # optimized revenue
    ama_performance, ama_performance_std = expectedperformance(lp, ama_params, num_samples=TEST_SAMPLES, α=0.0, require_optimal=true) # optimized revenue
    @show ama_revenue
    @show ama_std

    # print csv line consisting of seed, vcg_revenue, optimized_revenue
    println("*****************")
    (
        (method="reglp",mdp=string(mdptype),dist=dist_type,num_agents=num_agents, num_items=num_items, seed=seed, num_samples=num_samples, test_samples=TEST_SAMPLES,training_iters=num_training_iters,lr=lr,reg_strength=reg_strength,vcg_performance=vcg_performance,
            vcg_performance_std=vcg_performance_std, ama_performance=ama_performance, ama_performance_std=ama_performance_std,
            vcg_revenue=vcg_revenue, vcg_std=vcg_std, ama_revenue=ama_revenue, ama_std=ama_std, runtime=(end_time - start_time)),
        (ama=ama_params, vals=vals)
    )
end

function namedtuple_to_csv_line_str(nt)
    # print all entries of namedtuple as csv
    str = ""
    for (k, v) in pairs(nt)
        str *= string(v) * "\t"
    end
    str
end

@main function main(num_agents::Int64, num_items::Int64, num_samples::Int64, num_training_iters::Int64, start_seed::Int64, num_trials::Int64, mdptype_str::String="auction", lr::Float64=0.01, dist_type::String="uniform", reg_strength::Float64=0.01, γ::Float64=1.0)
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
    @show num_training_iters
    @show start_seed
    @show num_trials
    @show mdptype_str
    @show lr
    @show dist_type
    @show reg_strength

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
    csv_filename = "results_" * mdptype_str * "_" * dist_type * "_" * string(num_agents) * "_" * string(num_items) * "_" * string(num_samples) * "_" * string(num_training_iters) * "_" * string(start_seed) * "_" * string(num_trials) * "_lr" * string(lr) * "_reg" * string(reg_strength) * ".csv"
    for trial = 1:num_trials
        @show trial, mdptype
        start_seed += 1
        boostvals_filename = "boostvals_" * mdptype_str * dist_type * "_trial" * string(trial) * "_" * string(num_agents) * "_" * string(num_items) * "_" * string(num_samples) * "_" * string(num_training_iters) * "_" * string(start_seed) * "_lr" * string(lr) * "_reg" * string(reg_strength) * ".jld2"
        use_weights = dist_type == "asymmetric"
        results, boosts_and_vals = runtrial(num_agents, num_items, num_samples, num_training_iters, start_seed, mdptype, lr, reg_strength, dist_type, optimize_weights=use_weights, γ=γ)
        # save boosts and vals
        save(filedir * boostvals_filename, "boosts_and_vals", boosts_and_vals)
        push!(all_results, results)
    end

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
