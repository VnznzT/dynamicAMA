function task_state_and_action_list(n_agents::Int, n_items::Int)
    actions = 1:n_agents
    # round, then who each item is allocated to
    start_state = (1, fill(-1, n_items)...)
    auction_state_list = [start_state]
    states_to_process = [start_state]

    while !isempty(states_to_process)
        state = popfirst!(states_to_process)
        # if state[1] == n_items
        #     continue  # terminal state
        # end
        for action in actions
            if !((state[1]) == n_items)
                new_state = [state...]
                new_state[new_state[1]+1] = action
                new_state[1] += 1
                new_state_tuple = tuple(new_state...)
                push!(auction_state_list, new_state_tuple)
                push!(states_to_process, new_state_tuple)
            end
        end
    end

    return auction_state_list, collect(actions)
end

struct ScheduleMDP <: MDP
    n_agents::Int
    n_items::Int
    γ::Float64
    state_list::Vector{Tuple}
    action_list::Vector{Int}
    dist_type::String

    function ScheduleMDP(n_agents::Int, n_items::Int, γ::Float64, dist_type::String)
        state_list, action_list = task_state_and_action_list(n_agents, n_items)
        new(n_agents, n_items, γ, state_list, action_list, dist_type)
    end
end

function transition_probability(scheduler::ScheduleMDP, prev_state, prev_action, state)
    auction_state_list = scheduler.state_list
    auction_actions = scheduler.action_list
    n_items = scheduler.n_items
    @assert prev_state in auction_state_list
    @assert state in auction_state_list

    # otherwise we're in a non-terminal state
    prob = 1.0
    # if rounds not consecutive, no transition
    if prev_state[1] + 1 != state[1]
        prob *= 0.0
    end

    # if prev assignments do not agree, no transition
    if prev_state[2:prev_state[1]] != state[2:prev_state[1]]
        prob *= 0.0
    end

    # if prev_action is not the same as the next assignment, no transition
    if prev_action != state[prev_state[1]+1]
        prob *= 0.0
    end

    return prob
end

function reward_from_alloc(auction::ScheduleMDP, s, a, type_params)
    rewards = zeros(auction.n_agents)
    rewards[a] = -type_params[a, s[1]]
    rewards
end

function remaining_makespan(mdp::ScheduleMDP, agent_types, final_assignments)
    # gives remaining makespan at the last time step
    rem_makespan = zeros(mdp.n_agents)
    for task = 1:mdp.n_items
        rem_makespan[rem_makespan.>0] .-= 1
        assigned_agent = final_assignments[task]
        rem_makespan[assigned_agent] += agent_types[assigned_agent, task]
    end
    return max(maximum(rem_makespan), 0.0)
end


function makespan_from_sa(auction::ScheduleMDP, s, a, type_params)
    if s[1] < auction.n_items
        return 0.0
    end

    # otherwise we're in a terminal state, so calculate the makespan
    # concatenate the assignments with the action
    slen = length(s)
    assignments = [s[2:length(s)-1]..., a]
    makespan = remaining_makespan(auction, type_params, assignments)
    makespan
end

function sampletypes(mdp::ScheduleMDP, num_samples)
    if mdp.dist_type == "uniform"
        [3.0 * rand(mdp.n_agents, mdp.n_items) for i = 1:num_samples]
    elseif mdp.dist_type == "asymmetric"
        weights = [1.0 * i for i = 1:mdp.n_agents]
        # x = [reshape(weights, :, 1) .* (3.0 * randn(mdp.n_agents, mdp.n_items)) for i = 1:num_samples]
        [weights .* (3.0 * rand(mdp.n_agents, mdp.n_items)) for i = 1:num_samples]
    else
        error("unknown distribution type")
    end
end


function startstate(mdp::ScheduleMDP)
    (1, fill(-1, mdp.n_items)...)
end




function nonterminal(mdp::ScheduleMDP, s)
    s[1] <= mdp.n_items
end


const SCHEDULER_SENTINEL = 5
function counterfactualtype(mdp::ScheduleMDP, types, i)
    res = copy(types)
    res[i, :] .= SCHEDULER_SENTINEL
    res
end

function dmakespan_dx(auction::ScheduleMDP, x, types)
    dmakespan = zeros(size(x))
    for state_ind = 1:length(auction.state_list)
        for action_ind = 1:length(auction.action_list)
            dmakespan[state_ind, action_ind] = makespan_from_sa(auction, auction.state_list[state_ind], auction.action_list[action_ind], types)
        end
    end
    dmakespan
end
