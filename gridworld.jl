function grid_state_and_action_list(grid_size::Int)
    # possibly change actions to enum?
    actions = ["up", "down", "left", "right"]
    # create a list of tuples, which are the coordinates of the grid_size
    states_matrix = [(i, j) for i in 1:grid_size, j in 1:grid_size]
    # flatten state matrix into variable states
    states = vec(states_matrix)
    return states, actions
end

struct GridMDP <: MDP
    n_agents::Int
    n_items::Int  # this is in fact grid size
    γ::Float64
    state_list::Vector{Tuple{Int, Int}}
    action_list::Vector{String}
    dist_type::String

    function GridMDP(n_agents::Int, n_items::Int, γ::Float64, dist_type::String)
        state_list, action_list = grid_state_and_action_list(n_items)
        new(n_agents, n_items, γ, state_list, action_list, dist_type)
    end
end


function startstate(mdp::GridMDP)
    (1, 1)
end

function next_state(grid::GridMDP,prev_state,prev_action)
    # Up down moves y coordinate, left right moves x coordinate
    # If you hit a boundary you warp around to the other side

    # remember n_items is grid_size
    x, y = prev_state
    if prev_action == "up"
        next_state = (y==1 ? (x, grid.n_items) : (x, y-1))
    elseif prev_action == "down"
        next_state = (y==grid.n_items ? (x, 1) : (x, y+1))
    elseif prev_action == "left"
        next_state = (x==1 ? (grid.n_items, y) : (x-1, y))
    elseif prev_action == "right"
        next_state = (x==grid.n_items ? (1, y) : (x+1, y))
    else
        error("Unknown action")
    end
    return next_state
end


function transition_probability(grid::GridMDP, prev_state, prev_action, state)
    grid_state_list = grid.state_list
    grid_actions = grid.action_list
    #check if this is needed
    grid_size = grid.n_items
    
    @assert prev_state in grid_state_list
    @assert state in grid_state_list

    # We just calculate the next state given the prev_state and the action and then we check if it is the same as the state
    next_state_state = next_state(grid,prev_state,prev_action)
    # check if the next state is the same as the state
    return next_state_state == state ? 1 : 0
end

#called reward_from_alloc in lp_and_ama.jl so we use it here as well
function reward_from_alloc(grid::GridMDP, s, a, type_params)
    #TODO why do we have γ when we never use it?

    #get next state
    x,y = next_state(grid,s,a)

    # Directly compute rewards using array slicing
    rewards = [type_params[i,x, y] for i in 1:grid.n_agents]

    return rewards
end

# function nonterminal(mdp::AuctionMDP, s)
#     s[1] < mdp.n_items
# end


function counterfactualtype(mdp::GridMDP, types, i)
    # a type is a matrix of size grid_size x grid_size, i.e. one type per grid cell
    # we copy the types and change the type of agent i to have type 0 everywhere
    res = copy(types)
    res[i,:,:] = zeros(mdp.n_items, mdp.n_items)
    res
end

#TODO fix num_samples
function sampletypes(mdp::GridMDP, num_samples::Int64)
    types = Vector{Array{Float64, 3}}(undef, num_samples)
    if mdp.dist_type == "onecell"
        # this means the agent only has one cell where he has a random type between 0 and 1
        # we sample the cell randomly
        #nitems is gridsize!
        # types = zeros(num_samples, mdp.n_agents, mdp.n_items, mdp.n_items)

        # for i in 1:mdp.n_agents
        #     coords = rand(1:mdp.n_items, 2, num_samples)  # Draw coordinates of length num_samples
        #     x = coords[1, :]
        #     y = coords[2, :]
        #     for j in 1:num_samples
        #         types[i, x[j], y[j]] = rand()
        #     end
        # end
        # return types

        for j in 1:num_samples
            sample = zeros(mdp.n_agents, mdp.n_items, mdp.n_items)
            
            for i in 1:mdp.n_agents
                # sample x and y coordinates randomly but exclude the outcome 1,1. if that appears, resample
                x,y = rand(1:mdp.n_items, 2)
                while x == 1 && y == 1
                    x,y = rand(1:mdp.n_items, 2)
                end
                sample[i, x, y] = rand()
            end
            
            types[j] = sample
        end

    elseif mdp.dist_type == "uniform"
        for j in 1:num_samples
            # uniform at random
            sample = rand(mdp.n_agents, mdp.n_items, mdp.n_items)
            sample[:, 1, 1] .= 0.0
            types[j] = sample
        end
    else
        error("Unknown distribution type")
    end
    return types
end

function nonterminal(mdp::GridMDP, s)
    true
end