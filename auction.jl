function auction_state_and_action_list(n_agents::Int, n_items::Int)
    actions = 0:n_agents
    # round, then who each item is allocated to
    start_state = (0, fill(-1, n_items)...)
    auction_state_list = [start_state]
    states_to_process = [start_state]

    while !isempty(states_to_process)
        state = popfirst!(states_to_process)
        # if state[1] == n_items
        #     continue  # terminal state
        # end
        for action in actions
            if !((state[1] + 1) == n_items)
                new_state = [state...]
                new_state[1] += 1
                new_state[new_state[1]+1] = action
                new_state_tuple = tuple(new_state...)
                push!(auction_state_list, new_state_tuple)
                push!(states_to_process, new_state_tuple)
            end
        end
    end

    return auction_state_list, collect(actions)
end

struct AuctionMDP <: MDP
    n_agents::Int
    n_items::Int
    γ::Float64
    state_list::Vector{Tuple}
    action_list::Vector{Int}
    dist_type::String

    function AuctionMDP(n_agents::Int, n_items::Int, γ::Float64, dist_type::String)
        state_list, action_list = auction_state_and_action_list(n_agents, n_items)
        new(n_agents, n_items, γ, state_list, action_list, dist_type)
    end
end

function startstate(mdp::AuctionMDP)
    (0, fill(-1, mdp.n_items)...)
end

function transition_probability(auction::AuctionMDP, prev_state, prev_action, state)
    auction_state_list = auction.state_list
    auction_actions = auction.action_list
    n_items = auction.n_items
    @assert prev_state in auction_state_list
    @assert state in auction_state_list

    # otherwise we're in a non-terminal state
    prob = 1.0
    # if rounds not consecutive, no transition
    if prev_state[1] + 1 != state[1]
        prob *= 0.0
    end

    # if prev assignments do not agree, no transition
    if prev_state[2:prev_state[1]+1] != state[2:prev_state[1]+1]
        prob *= 0.0
    end

    # if prev_action is not the same as the next assignment, no transition
    if prev_action != state[prev_state[1]+2]
        prob *= 0.0
    end

    return prob
end


function reward_from_alloc(auction::AuctionMDP, s, a, type_params)
    n_items = auction.n_items
    rewards = zeros(auction.n_agents)
    if a != 0
        # if agent not already allocated, give them the reward
        if a ∉ s[2:end]
            rewards[a] = type_params[a]
        end
    end
    return rewards
end

function nonterminal(mdp::AuctionMDP, s)
    s[1] < mdp.n_items
end


function counterfactualtype(mdp::AuctionMDP, types, i)
    res = copy(types)
    res[i] = 0.0
    res
end

function sampletypes(mdp::AuctionMDP, num_samples)
    if mdp.dist_type == "uniform"
        [rand(mdp.n_agents) for _ in 1:num_samples]
    elseif mdp.dist_type == "asymmetric"
        weights = [1.0 / i for i in 1:mdp.n_agents]
        [weights .* rand(mdp.n_agents) for _ in 1:num_samples]
    else
        error("Unknown distribution type")
    end
end
