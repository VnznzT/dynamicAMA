abstract type MDP end

include("auction.jl")
include("schedule.jl")
include("matching.jl")
include("gridworld.jl")

struct AMAParams
    weights::Vector{Float64}
    boosts::Matrix{Float64}
end

struct MDPLinearProgram{T<:MDP}
    mdp::T
    model::Model
    x::Matrix{VariableRef}
    t::Matrix{VariableRef}

    function MDPLinearProgram(mdp::T) where {T<:MDP}
        #model = Model(() -> DiffOpt.diff_optimizer(CHOSEN_OPTIM))
        model = Model(() -> DiffOpt.diff_optimizer(optimizer_with_attributes(Mosek.Optimizer, "MSK_IPAR_NUM_THREADS" => 4)))
        #TODO check if this works
        set_attribute(model, MOI.Silent(), true)
        x = @variable(model, [1:length(mdp.state_list), 1:length(mdp.action_list)], lower_bound=0.0)
        t = @variable(model, [1:length(mdp.state_list), 1:length(mdp.action_list)])
        auction_state_list = mdp.state_list
        auction_actions = mdp.action_list
        n_agents = mdp.n_agents
        n_items = mdp.n_items

        for state_ind in 1:length(auction_state_list)
            constraint_lhs_expr = sum(x[state_ind, action_ind] for action_ind in 1:length(auction_actions))

            rhs_expr_terms = []
            for prev_state_ind in 1:length(auction_state_list)
                for prev_action_ind in 1:length(auction_actions)
                    tp = transition_probability(mdp, auction_state_list[prev_state_ind], auction_actions[prev_action_ind], auction_state_list[state_ind])
                    push!(rhs_expr_terms, mdp.γ * x[prev_state_ind, prev_action_ind] * tp)
                end
            end

            constraint_rhs_expr = sum(rhs_expr_terms)

            if auction_state_list[state_ind] == startstate(mdp)
                constraint_rhs_expr += 1.0
            end

            if nonterminal(mdp, auction_state_list[state_ind])
                @constraint(model, constraint_lhs_expr == constraint_rhs_expr)
            end
        end

        # entropy constraints
        for state_ind in 1:length(auction_state_list)
            for action_ind in 1:length(auction_actions)
                @constraint(model, [t[state_ind, action_ind], x[state_ind, action_ind], 1] in MOI.ExponentialCone())
            end
        end
        new{T}(mdp, model, x, t)
    end
end

# for testing purposes -- should be safe to use MDP above with α=0.0 for unregularized case.
# presolve was failing because of terminal states having reward
#duplicate
struct UnregMDP{T<:MDP}
    mdp::T
    model::Model
    x::Matrix{VariableRef}

    function UnregMDP(mdp::T) where {T<:MDP}
        model = Model(CHOSEN_OPTIM)
        set_attribute(model, MOI.Silent(), true)
        x = @variable(model, [1:length(mdp.state_list), 1:length(mdp.action_list)])
        auction_state_list = mdp.state_list
        auction_actions = mdp.action_list
        n_agents = mdp.n_agents
        n_items = mdp.n_items

        @constraint(model, x .>= 0.0)

        for state_ind in 1:length(auction_state_list)
            constraint_lhs_expr = sum(x[state_ind, action_ind] for action_ind in 1:length(auction_actions))

            rhs_expr_terms = []
            for prev_state_ind in 1:length(auction_state_list)
                for prev_action_ind in 1:length(auction_actions)
                    tp = transition_probability(mdp, auction_state_list[prev_state_ind], auction_actions[prev_action_ind], auction_state_list[state_ind])
                    push!(rhs_expr_terms, mdp.γ * x[prev_state_ind, prev_action_ind] * tp)
                end
            end

            constraint_rhs_expr = sum(rhs_expr_terms)

            if auction_state_list[state_ind] == startstate(mdp)
                constraint_rhs_expr += 1.0
            end

            if nonterminal(mdp, auction_state_list[state_ind])
                @constraint(model, constraint_lhs_expr == constraint_rhs_expr)
            end
        end

        # entropy constraints
        new{T}(mdp, model, x)
    end
end

function evalauction!(auctionlp::MDPLinearProgram{T}, types, ama, α) where {T<:MDP}

    boosts = ama.boosts

    rew = α * sum(auctionlp.t)
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

    #This was just to check for gridworld that the solution is feasible
    #tolerance = 1e-6
    #@assert all(value.(auctionlp.x) .≥ -tolerance)
end

function asw(auction::MDP, x, types, ama)
    boosts = ama.boosts
    total_asw = 0.0
    for state_ind = 1:length(auction.state_list)
        for action_ind = 1:length(auction.action_list)
            total_asw += x[state_ind, action_ind] *
                         (sum(ama.weights .* reward_from_alloc(auction, auction.state_list[state_ind], auction.action_list[action_ind], types)) +
                          boosts[state_ind, action_ind])
        end
    end
    total_asw
end

function calcrevenue(lp::MDPLinearProgram, types, ama, α; require_optimal=false)
    boosts = ama.boosts
    evalauction!(lp, types, ama, α)
    if JuMP.termination_status(lp.model) != MOI.OPTIMAL
        if !require_optimal
            println("WARNING: NOT OPTIMAL")
        else
            @show types, ama, α
            error("failed to be optimal when required")
        end
    end
    main_x = value.(lp.x)
    asw_main_x = asw(lp.mdp, main_x, types, ama)
    sw_main_x = sw(lp.mdp, main_x, types)

    rev = sw_main_x

    for i = 1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        evalauction!(lp, counterfactual_types, ama, α)
        if JuMP.termination_status(lp.model) != MOI.OPTIMAL
            if !require_optimal
                println("WARNING: NOT OPTIMAL")
            else
                @show types, ama, α
                error("failed to be optimal when required")
            end
        end
        asw_counterfactual_x = asw(lp.mdp, value.(lp.x), counterfactual_types, ama)
        rev += (asw_counterfactual_x - asw_main_x) / ama.weights[i]
    end
    rev
end

function sw(auction::MDP, x, types)
    total_sw = 0.0
    for state_ind = 1:length(auction.state_list)
        for action_ind = 1:length(auction.action_list)
            total_sw += x[state_ind, action_ind] * sum(reward_from_alloc(auction, auction.state_list[state_ind], auction.action_list[action_ind], types))
        end
    end
    total_sw
end

function expectedrevenue(lp::MDPLinearProgram, ama; num_samples=1000, α=0.01, require_optimal=false)
    # types = rand(lp.mdp.n_agents, num_samples)
    types = sampletypes(lp.mdp, num_samples)
    all_revs = Array{Float64}(undef, num_samples)
    
    for i = 1:num_samples
        all_revs[i] = calcrevenue(lp, types[i], ama, α, require_optimal=require_optimal)
    end

    (sum(all_revs) / num_samples, std(all_revs))
end

function calcmakespan(lp::MDPLinearProgram{ScheduleMDP}, types, ama, α; require_optimal=false)
    evalauction!(lp, types, ama, α)
    if JuMP.termination_status(lp.model) != MOI.OPTIMAL
        if !require_optimal
            println("WARNING: NOT OPTIMAL")
        else
            @show types, ama, α
            error("failed to be optimal when required")
        end
    end
    x = value.(lp.x)
    total_makespan = 0.0
    for state_ind = 1:length(lp.mdp.state_list)
        for action_ind = 1:length(lp.mdp.action_list)
            makespan = makespan_from_sa(lp.mdp, lp.mdp.state_list[state_ind], lp.mdp.action_list[action_ind], types)
            total_makespan += x[state_ind, action_ind] * makespan
        end
    end
    total_makespan
end

expectedperformance(lp::MDPLinearProgram{AuctionMDP}, ama; num_samples=1000, α=0.01, require_optimal=false) = expectedrevenue(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)
expectedperformance(lp::MDPLinearProgram{MatchingMDP}, ama; num_samples=1000, α=0.01, require_optimal=false) = expectedrevenue(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)
expectedperformance(lp::MDPLinearProgram{GridMDP},ama; num_samples=1000, α=0.01,require_optimal=false) = expectedrevenue(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)

function expectedperformance(lp::MDPLinearProgram{ScheduleMDP}, ama; num_samples=1000, α=0.01, require_optimal=false)
    mkspan, std = expectedmakespan(lp, ama, num_samples=num_samples, α=α, require_optimal=require_optimal)
    return (-mkspan, std)
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

function dsw_dx(auction::MDP, x, types)
    dsw = zeros(size(x))
    for state_ind = 1:length(auction.state_list)
        for action_ind = 1:length(auction.action_list)
            dsw[state_ind, action_ind] = sum(reward_from_alloc(auction, auction.state_list[state_ind], auction.action_list[action_ind], types))
        end
    end
    dsw
end


function dasw_dx(auction::MDP, x, types, ama)
    boosts = ama.boosts
    dasw = zeros(size(x))
    for state_ind = 1:length(auction.state_list)
        for action_ind = 1:length(auction.action_list)
            dasw[state_ind, action_ind] = sum(ama.weights .* reward_from_alloc(auction, auction.state_list[state_ind], auction.action_list[action_ind], types)) + boosts[state_ind, action_ind]
        end
    end
    dasw
end
