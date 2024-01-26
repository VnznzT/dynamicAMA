function matching_state_and_action_list(n_agents::Int, n_items::Int)
    actions = 0:n_agents

    start_state = (-1, fill(-2, n_agents)...)


    auction_state_list = Set([start_state])
    states_to_process = [start_state]

    while !isempty(states_to_process)
        state = popfirst!(states_to_process)
        # if state[1] == n_items
        #     continue  # terminal state
        # end
        # an agent arrives, and we match
        for (action, arriving_agent) in Iterators.product(actions, 0:n_agents)
            new_state = [state...]
            new_state[1] += 1


            # -2 is not arrived
            # -1 is matched
            # otherwise it is arrival round
            # if we're not in a terminal state
            if !(state[1] + 1 == n_items)

                if action > 0
                    # we get a next state when the match is valid, AND the arriving agent has not yet arrived
                    agent_to_match_is_matchable = state[1+action] >= 0  # arrived but not matched
                    arriving_agent_has_not_arrived = (state[1+arriving_agent] == -2) || (arriving_agent == 0)


                    if agent_to_match_is_matchable && arriving_agent_has_not_arrived
                        new_state[1+action] = -1
                        if arriving_agent > 0
                            new_state[1+arriving_agent] = new_state[1]
                        end
                        push!(auction_state_list, tuple(new_state...))
                        push!(states_to_process, tuple(new_state...))
                    end
                else
                    arriving_agent_has_not_arrived = (state[1+arriving_agent] == -2) || (arriving_agent == 0)
                    if arriving_agent_has_not_arrived
                        if arriving_agent > 0
                            new_state[1+arriving_agent] = new_state[1]
                        end
                        push!(auction_state_list, tuple(new_state...))
                        push!(states_to_process, tuple(new_state...))
                    end
                end
            end
        end
    end

    sorted_state_list = sort(collect(auction_state_list))

    return sorted_state_list, collect(actions)
end

struct MatchingMDP <: MDP
    n_agents::Int
    n_items::Int
    γ::Float64
    state_list::Vector{Tuple}
    action_list::Vector{Int}
    dist_type::String

    function MatchingMDP(n_agents::Int, n_items::Int, γ::Float64, dist_type::String)
        state_list, action_list = matching_state_and_action_list(n_agents, n_items)
        new(n_agents, n_items, γ, state_list, action_list, dist_type)
    end
end

function startstate(mdp::MatchingMDP)
    (-1, fill(-2, mdp.n_agents)...)
end

function nonterminal(mdp::MatchingMDP, state)
    state[1] != mdp.n_items
end

# type consists of value + time length
function counterfactualtype(mdp::MatchingMDP, types, i)
    res = copy(types)
    res[i, 1] = 0.0
    res
end

function sampletypes(mdp::MatchingMDP, num_samples)
    if mdp.dist_type == "uniform"
        [hcat(rand(mdp.n_agents), Float64.(rand(1:5, mdp.n_agents))) for _ in 1:num_samples]
    elseif mdp.dist_type == "asymmetric"
        weights = [1.0 / i for i = 1:mdp.n_agents]
        [hcat(weights .* rand(mdp.n_agents), Float64.(rand(1:5, mdp.n_agents))) for _ in 1:num_samples]
    else
        error("unknown distribution type")
    end
end

function transition_probability(mdp::MatchingMDP, prev_state, action, state)
    # we should first check if the transition is even valid in terms of the action
    @assert prev_state in mdp.state_list
    @assert action in mdp.action_list
    @assert state in mdp.state_list

    remaining_unarrived_agents = sum(prev_state[2:end] .== -2)

    # note this is also the probability that nobody shows up this round, hence +1
    agent_arrival_prob = 1.0 / (remaining_unarrived_agents + 1)

    prob = agent_arrival_prob

    # if rounds not consecutive, no transition
    if prev_state[1] + 1 != state[1]
        prob *= 0.0
    end

    # all agents should be the same except at most 1 arrival, which should be in this round
    # and if action > 0, 

    arrival_count = 0
    match_count = 0

    for i = 2:length(state)
        if prev_state[i] == -2
            if state[i] == -1 # we matched an unarrived agent
                prob *= 0.0
            elseif state[i] >= 0  # ensure that the agent arrived in this round
                if state[i] != state[1]
                    prob *= 0.0
                end


                # arrivals happen with probability
                arrival_count += 1
            end
        elseif prev_state[i] == -1
            if state[i] != -1
                prob *= 0.0  # matched agents stay matched
            end
        elseif prev_state[i] >= 0
            if state[i] == -2
                prob *= 0.0  # arrived agents can't be unarrived
            elseif state[i] == -1
                if action != i - 1
                    prob *= 0.0  # it wasn't this agent who was matched
                end
                match_count += 1
            elseif state[i] == prev_state[i]
                if action == i - 1
                    prob *= 0.0 # this agent WAS matched, so this is invalid transition
                end
            elseif state[i] != prev_state[i]
                prob *= 0.0  # agents can't change their arrival round
            end
        end
    end

    if arrival_count > 1
        prob *= 0.0
    end

    if match_count > 1
        prob *= 0.0
    end
    prob
end

function reward_from_alloc(mdp::MatchingMDP, s, a, type_params)
    rewards = zeros(mdp.n_agents)
    if s[1] >= mdp.n_items
        # terminal state
        return rewards
    end
    if a != 0
        # chosen agent has arrived (arrival round >= 0)
        # and is matchable (current_round - arrival_round <= type_params[a, 2])
        arrival_round = s[1+a]
        current_round = s[1]
        if (arrival_round >= 0) && (current_round - arrival_round <= type_params[a, 2])
            rewards[a] = type_params[a, 1]
        end
    end
    rewards
end


function test_transitions(mdp)
    for prev_state in mdp.state_list
        for action in mdp.action_list
            probs = [transition_probability(mdp, prev_state, action, state) for state in mdp.state_list]
            @show prev_state, action, probs, sum(probs)
        end
    end
end