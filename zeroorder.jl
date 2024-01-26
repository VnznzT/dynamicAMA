
using JuMP, DiffOpt, LinearAlgebra, MosekTools, COSMO, ProgressMeter, Random, Statistics, FileIO, Dates, Comonicon
const CHOSEN_OPTIM = Mosek.Optimizer
# take AMA parameters
# solve LP into yk
# perturb slightly with perturbation uj
# solve LP into ykj
# calculate dj = (ykj - yj) / magnitude
# first gradient term is just grad
# second term is average of < dj, grad_y f(x,yk) > * uj
include("lp_and_ama.jl")
function namedtuple_to_csv_line_str(nt)
    # print all entries of namedtuple as csv
    str = ""
    for (k, v) in pairs(nt)
        str *= string(v) * "\t"
    end
    str
end

const TEST_SAMPLES=10000


function zeroorder_jac_makespan(lp::MDPLinearProgram, main_x, types, ama; num_perturbations=10, α=0.0, noise_magnitude=0.01)
    normal_samples = [randn(size(ama.boosts)) for _ in 1:num_perturbations]
    deltas = []
    for j=1:num_perturbations
        new_boost = ama.boosts + normal_samples[j] * noise_magnitude
        new_ama = AMAParams(ama.weights, new_boost)
        evalauction!(lp, types, new_ama, α)
        new_x = value.(lp.x)
        # first order estimate of delta
        push!(deltas, (new_x - main_x) / noise_magnitude)
    end
    average_jacobian_term = zeros(size(ama.boosts))
    grady =  dmakespan_dx(lp.mdp, main_x, types)
    for j=1:num_perturbations
        average_jacobian_term .+= sum(deltas[j] .* grady)*normal_samples[j]
    end
    average_jacobian_term ./= num_perturbations
    average_jacobian_term
end

function zeroorder_jac_makespan_wb(lp::MDPLinearProgram, main_x, types, ama; num_perturbations=10, α=0.0, noise_magnitude=0.01)
    # although separate in code, treat both w and b perturbations as one big u_{k, j} vector per equations in PZOBO paper
    normal_samples = [randn(size(ama.boosts)) for _ in 1:num_perturbations]
    normal_samples_w = [randn(size(ama.weights)) for _ in 1:num_perturbations]
    deltas = []
    for j=1:num_perturbations
        new_boost = ama.boosts + normal_samples[j] * noise_magnitude
        new_weights = ama.weights + normal_samples_w[j] * noise_magnitude
        new_ama = AMAParams(new_weights, new_boost)
        evalauction!(lp, types, new_ama, α)
        new_x = value.(lp.x)
        # first order estimate of delta
        push!(deltas, (new_x - main_x) / noise_magnitude)
    end
    average_jacobian_term_b = zeros(size(ama.boosts))
    average_jacobian_term_w = zeros(size(ama.weights))
    grady =  dmakespan_dx(lp.mdp, main_x, types)
    for j=1:num_perturbations
        inner_product_term = sum(deltas[j] .* grady)
        average_jacobian_term_b .+= inner_product_term * normal_samples[j]
        average_jacobian_term_w .+= inner_product_term * normal_samples_w[j]
    end
    average_jacobian_term_b ./= num_perturbations
    average_jacobian_term_w ./= num_perturbations
    (average_jacobian_term_w, average_jacobian_term_b)
end

function zeroorder_jac_factual_rev_b(lp::MDPLinearProgram, main_x, types, ama; num_perturbations=10, α=0.0, noise_magnitude=0.01)
    normal_samples = [randn(size(ama.boosts)) for _ in 1:num_perturbations]
    deltas = []
    for j=1:num_perturbations
        new_boost = ama.boosts + normal_samples[j] * noise_magnitude
        new_ama = AMAParams(ama.weights, new_boost)
        evalauction!(lp, types, new_ama, α)
        new_x = value.(lp.x)
        # first order estimate of delta
        push!(deltas, (new_x - main_x) / noise_magnitude)
    end
    average_jacobian_term = zeros(size(ama.boosts))
    grady = zeros(size(ama.boosts))
    for state_ind = 1:length(lp.mdp.state_list)
        for action_ind = 1:length(lp.mdp.action_list)
            rew = reward_from_alloc(lp.mdp, lp.mdp.state_list[state_ind], lp.mdp.action_list[action_ind], types)

            # gradient correct per ...
            grady[state_ind, action_ind] = sum([(1.0/ama.weights[i])*-(sum(ama.weights .* rew) + ama.boosts[state_ind, action_ind]) for i=1:lp.mdp.n_agents]) + sum(rew)
        end
    end
    for j=1:num_perturbations
        average_jacobian_term .+= sum(deltas[j] .* grady)*normal_samples[j]
    end
    average_jacobian_term ./= num_perturbations
    average_jacobian_term
end

function zeroorder_jac_factual_rev_wb(lp::MDPLinearProgram, main_x, types, ama; num_perturbations=10, α=0.0, noise_magnitude=0.01)
    # although separate in code, treat both w and b perturbations as one big u_{k, j} vector per equations in PZOBO paper
    normal_samples = [randn(size(ama.boosts)) for _ in 1:num_perturbations]
    normal_samples_w = [randn(size(ama.weights)) for _ in 1:num_perturbations]
    deltas = []
    for j=1:num_perturbations
        new_boost = ama.boosts + normal_samples[j] * noise_magnitude
        new_weights = ama.weights + normal_samples_w[j] * noise_magnitude
        new_ama = AMAParams(new_weights, new_boost)
        evalauction!(lp, types, new_ama, α)
        new_x = value.(lp.x)
        # first order estimate of delta
        push!(deltas, (new_x - main_x) / noise_magnitude)
    end
    average_jacobian_term_b = zeros(size(ama.boosts))
    average_jacobian_term_w = zeros(size(ama.weights))
    grady = zeros(size(ama.boosts))
    for state_ind = 1:length(lp.mdp.state_list)
        for action_ind = 1:length(lp.mdp.action_list)
            rew = reward_from_alloc(lp.mdp, lp.mdp.state_list[state_ind], lp.mdp.action_list[action_ind], types)

            # gradient correct per anonymous author
            grady[state_ind, action_ind] = sum([(1.0/ama.weights[i])*-(sum(ama.weights .* rew) + ama.boosts[state_ind, action_ind]) for i=1:lp.mdp.n_agents]) + sum(rew)
        end
    end
    for j=1:num_perturbations
        inner_product_term = sum(deltas[j] .* grady)
        average_jacobian_term_b .+= inner_product_term * normal_samples[j]
        average_jacobian_term_w .+= inner_product_term * normal_samples_w[j]
    end
    average_jacobian_term_b ./= num_perturbations
    average_jacobian_term_w ./= num_perturbations
    (average_jacobian_term_w, average_jacobian_term_b)
end

function zeroorder_jac_counterfactual_rev_wb(lp::MDPLinearProgram, main_x, types, ama, i; num_perturbations=10, α=0.0, noise_magnitude=0.01)
    # although separate in code, treat both w and b perturbations as one big u_{k, j} vector per equations in PZOBO paper
    types = counterfactualtype(lp.mdp, types, i)
    normal_samples = [randn(size(ama.boosts)) for _ in 1:num_perturbations]
    normal_samples_w = [randn(size(ama.weights)) for _ in 1:num_perturbations]
    deltas = []
    for j=1:num_perturbations
        new_boost = ama.boosts + normal_samples[j] * noise_magnitude
        new_weights = ama.weights + normal_samples_w[j] * noise_magnitude
        new_ama = AMAParams(new_weights, new_boost)
        evalauction!(lp, types, new_ama, α)
        new_x = value.(lp.x)
        # first order estimate of delta
        push!(deltas, (new_x - main_x) / noise_magnitude)
    end
    average_jacobian_term_b = zeros(size(ama.boosts))
    average_jacobian_term_w = zeros(size(ama.weights))
    grady = zeros(size(ama.boosts))
    for state_ind = 1:length(lp.mdp.state_list)
        for action_ind = 1:length(lp.mdp.action_list)
            # rew = reward_from_alloc(lp.mdp, lp.mdp.state_list[state_ind], lp.mdp.action_list[action_ind], types)

            # gradient correct per ...
            grady[state_ind, action_ind] = (1.0/ama.weights[i])*sum(ama.weights .* reward_from_alloc(lp.mdp, lp.mdp.state_list[state_ind], lp.mdp.action_list[action_ind], types)) + ama.boosts[state_ind, action_ind]
        end
    end
    for j=1:num_perturbations
        inner_product_term = sum(deltas[j] .* grady)
        average_jacobian_term_b .+= inner_product_term * normal_samples[j]
        average_jacobian_term_w .+= inner_product_term * normal_samples_w[j]
    end
    average_jacobian_term_b ./= num_perturbations
    average_jacobian_term_w ./= num_perturbations
    (average_jacobian_term_w, average_jacobian_term_b)
end

function zeroorder_jac_counterfactual_rev_b(lp::MDPLinearProgram, main_x, types, ama, i; num_perturbations=10, α=0.0, noise_magnitude=0.01)
    types = counterfactualtype(lp.mdp, types, i)
    normal_samples = [randn(size(ama.boosts)) for _ in 1:num_perturbations]
    deltas = []
    for j=1:num_perturbations
        new_boost = ama.boosts + normal_samples[j] * noise_magnitude
        new_ama = AMAParams(ama.weights, new_boost)
        evalauction!(lp, types, new_ama, α)
        new_x = value.(lp.x)
        # first order estimate of delta
        push!(deltas, (new_x - main_x) / noise_magnitude)
    end
    average_jacobian_term = zeros(size(ama.boosts))

    # 1/w_i * (sum_{j != i} w_j r_j(s, a) + boost(s, a))
    grady = zeros(size(ama.boosts))
    for state_ind = 1:length(lp.mdp.state_list)
        for action_ind = 1:length(lp.mdp.action_list)
            # CHECK SIGNS
            grady[state_ind, action_ind] = (1.0/ama.weights[i])*sum(ama.weights .* reward_from_alloc(lp.mdp, lp.mdp.state_list[state_ind], lp.mdp.action_list[action_ind], types)) + ama.boosts[state_ind, action_ind]
        end
    end

    for j=1:num_perturbations
        average_jacobian_term .+= sum(deltas[j] .* grady)*normal_samples[j]
    end
    average_jacobian_term ./= num_perturbations
    average_jacobian_term
end

function zeroorder_makespangradb(lp::MDPLinearProgram, types, ama; num_perturb=10, α=0.0, noise_magnitude=0.01)
    # not implemented
    # gradient through leader objective alone, is 0
    evalauction!(lp, types, ama, α)
    main_x = value.(lp.x)

    # zero-order estimate of gradient through best response
    average_jacobian_term = zeroorder_jac_makespan(lp, main_x, types, ama, num_perturbations=num_perturb, α=α, noise_magnitude=noise_magnitude)
    # double check this
    average_jacobian_term
end

function zeroorder_makespangradwb(lp::MDPLinearProgram, types, ama; num_perturb=10, α=0.0, noise_magnitude=0.01)
    # not implemented
    # gradient through leader objective alone, is 0
    evalauction!(lp, types, ama, α)
    main_x = value.(lp.x)

    # zero-order estimate of gradient through best response
    average_jacobian_term_w, average_jacobian_term_b = zeroorder_jac_makespan_wb(lp, main_x, types, ama, num_perturbations=num_perturb, α=α, noise_magnitude=noise_magnitude)
    # double check this
    (average_jacobian_term_w, average_jacobian_term_b)
end

function zeroorder_revenuegradb(lp::MDPLinearProgram, types, ama; num_perturb=10, α=0.0, noise_magnitude=0.01)
    # compute a zero-order estimate of the derivative of revenue wrt boosts, for one fixed 
    # first solve unperturbed
    evalauction!(lp, types, ama, α)

    leader_grad_term = zeros(size(ama.boosts))
    main_x = value.(lp.x)
    counterfactual_solns = []

    for i=1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        evalauction!(lp, counterfactual_types, ama, α)
        counterfactual_x = value.(lp.x)
        push!(counterfactual_solns, counterfactual_x)
        leader_grad_term += (counterfactual_x - main_x) / ama.weights[i]
    end

    # factual jacobian grad term
    average_jacobian_term = zeroorder_jac_factual_rev_b(lp, main_x, types, ama, num_perturbations=num_perturb, α=α, noise_magnitude=noise_magnitude)

    counterfactual_jacobian_estimates = []
    for i=1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        push!(counterfactual_jacobian_estimates, zeroorder_jac_counterfactual_rev_b(lp, counterfactual_solns[i], counterfactual_types, ama, i, num_perturbations=num_perturb, α=α, noise_magnitude=noise_magnitude))
    end

    # CHECK SIGNS
    (leader_grad_term + average_jacobian_term + sum(counterfactual_jacobian_estimates))
end

function zeroorder_revenuegrad_wb(lp::MDPLinearProgram, types, ama; num_perturb=10, α=0.0, noise_magnitude=0.01)
    # compute a zero-order estimate of the derivative of revenue wrt boosts, for one fixed 
    # first solve unperturbed
    evalauction!(lp, types, ama, α)


    leader_grad_term_b = zeros(size(ama.boosts))
    leader_grad_term_w = zeros(size(ama.weights))
    main_x = value.(lp.x)
    # calculate asw
    main_asw = asw(lp.mdp, main_x, types, ama)
    counterfactual_solns = []

    counterfactual_asws = zeros(lp.mdp.n_agents)
    for i=1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        evalauction!(lp, counterfactual_types, ama, α)
        counterfactual_x = value.(lp.x)
        push!(counterfactual_solns, counterfactual_x)
        counterfactual_asws[i] = asw(lp.mdp, counterfactual_x, counterfactual_types, ama)
        leader_grad_term_b += (counterfactual_x - main_x) / ama.weights[i]
    end

    leader_grad_term_w .+= (main_asw ./ (ama.weights .^ 2)) 


    # factual_sw_by_agent[j] gives j's reward under main policy
    factual_sw_by_agent = zeros(lp.mdp.n_agents)

    # counterfactual_sw_by_agent[i, j] gives j's reward under -i
    counterfactual_sw_by_agent = zeros(lp.mdp.n_agents, lp.mdp.n_agents)

    for state=1:length(lp.mdp.state_list)
        for action=1:length(lp.mdp.action_list)
            factual_sw_by_agent .+= main_x[state, action] .* reward_from_alloc(lp.mdp, lp.mdp.state_list[state], lp.mdp.action_list[action], types)
            for i=1:lp.mdp.n_agents
                counterfactual_types = counterfactualtype(lp.mdp, types, i)
                counterfactual_sw_by_agent[i, :] .+= counterfactual_solns[i][state, action] .* reward_from_alloc(lp.mdp, lp.mdp.state_list[state], lp.mdp.action_list[action], counterfactual_types)
            end
        end
    end

    sumrecip = sum( 1.0 ./ ama.weights )

    leader_grad_term_w .-= sumrecip .* factual_sw_by_agent

    for j=1:lp.mdp.n_agents
        leader_grad_term_w[j] -= counterfactual_asws[j] / (ama.weights[j] ^ 2)
        for i=1:lp.mdp.n_agents
            if i != j
                leader_grad_term_w[j] += counterfactual_sw_by_agent[i, j] / ama.weights[i]
            end
        end
    end

    # factual jacobian grad term
    average_jacobian_term_w, average_jacobian_term_b = zeroorder_jac_factual_rev_wb(lp, main_x, types, ama, num_perturbations=num_perturb, α=α, noise_magnitude=noise_magnitude)

    counterfactual_jacobian_estimates_b = []
    counterfactual_jacobian_estimates_w = []
    for i=1:lp.mdp.n_agents
        counterfactual_types = counterfactualtype(lp.mdp, types, i)
        counterfactual_jac_est_w, counterfactual_jac_est_b = zeroorder_jac_counterfactual_rev_wb(lp, counterfactual_solns[i], counterfactual_types, ama, i, num_perturbations=num_perturb, α=α, noise_magnitude=noise_magnitude)
        push!(counterfactual_jacobian_estimates_b, counterfactual_jac_est_b)
        push!(counterfactual_jacobian_estimates_w, counterfactual_jac_est_w)
    end

    ( leader_grad_term_w + average_jacobian_term_w + sum(counterfactual_jacobian_estimates_w)  , leader_grad_term_b + average_jacobian_term_b + sum(counterfactual_jacobian_estimates_b))
end

function zeroorder_expectedrevenuegrad(lp::MDPLinearProgram{AuctionMDP}, ama::AMAParams; num_samples=10, num_perturb=10, α=0.00, noise_magnitude=0.01)
    # zero-order estimated gradient through follower best-response map

    types = sampletypes(lp.mdp, num_samples)
    rev_grad_b = zeros(size(ama.boosts))
    
    for i=1:num_samples
        rev_grad_b .+= zeroorder_revenuegradb(lp, types[i], ama, α=α, num_perturb=num_perturb, noise_magnitude=noise_magnitude)
    end
    rev_grad_b ./= num_samples
    rev_grad_b
end

function zeroorder_expectedrevenuegrad(lp::MDPLinearProgram{GridMDP}, ama::AMAParams; num_samples=10, num_perturb=10, α=0.00, noise_magnitude=0.01)
    # zero-order estimated gradient through follower best-response map

    types = sampletypes(lp.mdp, num_samples)
    rev_grad_b = zeros(size(ama.boosts))
    
    for i=1:num_samples
        rev_grad_b .+= zeroorder_revenuegradb(lp, types[i], ama, α=α, num_perturb=num_perturb, noise_magnitude=noise_magnitude)
    end
    rev_grad_b ./= num_samples
    rev_grad_b
end

function zeroorder_expectedrevenuegrad_wb(lp::MDPLinearProgram{AuctionMDP}, ama::AMAParams; num_samples=10, num_perturb=10, α=0.00, noise_magnitude=0.01)
    # zero-order estimated gradient through follower best-response map

    types = sampletypes(lp.mdp, num_samples)
    rev_grad_w, rev_grad_b = zeroorder_revenuegrad_wb(lp, types[1], ama, α=α, num_perturb=num_perturb, noise_magnitude=noise_magnitude)
    
    for i=2:num_samples
        rev_grad_w_i, rev_grad_b_i = zeroorder_revenuegrad_wb(lp, types[i], ama, α=α, num_perturb=num_perturb, noise_magnitude=noise_magnitude)
        rev_grad_w .+= rev_grad_w_i
        rev_grad_b .+= rev_grad_b_i
    end
    rev_grad_w ./= num_samples
    rev_grad_b ./= num_samples
    (rev_grad_w, rev_grad_b)
end

function zeroorder_expectedmakespan_grad(lp::MDPLinearProgram{ScheduleMDP}, ama::AMAParams; num_samples=10, num_perturb=10, α=0.00, noise_magnitude=0.01)
    # zero-order estimated gradient through follower best-response map

    types = sampletypes(lp.mdp, num_samples)
    makespan_grad_b = zeros(size(ama.boosts))
    
    for i=1:num_samples
        makespan_grad_b .+= zeroorder_makespangradb(lp, types[i], ama, α=α, num_perturb=num_perturb, noise_magnitude=noise_magnitude)
    end
    makespan_grad_b ./= num_samples
    makespan_grad_b
end

function zeroorder_expectedmakespan_grad_wb(lp::MDPLinearProgram{ScheduleMDP}, ama::AMAParams; num_samples=10, num_perturb=10, α=0.00, noise_magnitude=0.01)
    # zero-order estimated gradient through follower best-response map

    types = sampletypes(lp.mdp, num_samples)
    makespan_grad_w, makespan_grad_b = zeroorder_makespangradwb(lp, types[1], ama, α=α, num_perturb=num_perturb, noise_magnitude=noise_magnitude)
    
    for i=2:num_samples
        makespan_grad_w_i, makespan_grad_b_i = zeroorder_makespangradwb(lp, types[i], ama, α=α, num_perturb=num_perturb, noise_magnitude=noise_magnitude)
        makespan_grad_w .+= makespan_grad_w_i
        makespan_grad_b .+= makespan_grad_b_i
    end
    makespan_grad_w ./= num_samples
    makespan_grad_b ./= num_samples
    (makespan_grad_w, makespan_grad_b)
end

function optimize_boosts!(lp::MDPLinearProgram{AuctionMDP}, ama; num_samples=10, num_perturb=10, α=0.00, lr=0.01, noise_magnitude=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters

        # zero-order estimated gradient through follower best-response map
        est_grad = zeroorder_expectedrevenuegrad(lp, ama, num_samples=num_samples, noise_magnitude=noise_magnitude, num_perturb=num_perturb, α=α)
        ama.boosts .+= lr * (est_grad)
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
    end
    (ama, vals)
end

function optimize_boosts!(lp::MDPLinearProgram{GridMDP}, ama; num_samples=10, num_perturb=10, α=0.00, lr=0.01, noise_magnitude=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters

        # zero-order estimated gradient through follower best-response map
        est_grad = zeroorder_expectedrevenuegrad(lp, ama, num_samples=num_samples, noise_magnitude=noise_magnitude, num_perturb=num_perturb, α=α)
        ama.boosts .+= lr * (est_grad)
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
    end
    (ama, vals)
end

function optimize_weights_and_boosts!(lp::MDPLinearProgram{AuctionMDP}, ama; num_samples=10, num_perturb=10, α=0.00, lr=0.01, noise_magnitude=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters

        # zero-order estimated gradient through follower best-response map
        est_grad_w, est_grad_b = zeroorder_expectedrevenuegrad_wb(lp, ama, num_samples=num_samples, noise_magnitude=noise_magnitude, num_perturb=num_perturb, α=α)
        ama.weights .+= lr * (est_grad_w)
        ama.boosts .+= lr * (est_grad_b)
        vals[i], _ = expectedrevenue(lp, ama, num_samples=num_samples, α=α)
    end
    (ama, vals)
end

function optimize_boosts!(lp::MDPLinearProgram{ScheduleMDP}, ama; num_samples=10, num_perturb=10, α=0.00, lr=0.01, noise_magnitude=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters

        # zero-order estimated gradient through follower best-response map
        est_grad = zeroorder_expectedmakespan_grad(lp, ama, num_samples=num_samples, noise_magnitude=noise_magnitude, num_perturb=num_perturb, α=α)
        ama.boosts .-= lr * (est_grad)
        vals[i], _ = expectedmakespan(lp, ama, num_samples=num_samples, α=α)
    end
    (ama, vals)
end

function optimize_weights_and_boosts!(lp::MDPLinearProgram{ScheduleMDP}, ama; num_samples=10, num_perturb=10, α=0.00, lr=0.01, noise_magnitude=0.01, num_iters=100)
    vals = Array{Float64}(undef, num_iters)
    print("start")
    @showprogress for i = 1:num_iters

        # zero-order estimated gradient through follower best-response map
        est_grad_w, est_grad_b = zeroorder_expectedmakespan_grad_wb(lp, ama, num_samples=num_samples, noise_magnitude=noise_magnitude, num_perturb=num_perturb, α=α)
        ama.weights .-= lr * (est_grad_w)
        ama.boosts .-= lr * (est_grad_b)
        vals[i], _ = expectedmakespan(lp, ama, num_samples=num_samples, α=α)
    end
    (ama, vals)
end

function runtrial(num_agents, num_items, num_samples, num_perturb, num_training_iters, seed, mdptype::Type{T}, lr, noise_magnitude, dist_type; optimize_weights=false, γ =1.0) where {T<:MDP}
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
   

    start_time = Dates.now()
    if optimize_weights
        (ama_params, vals) = optimize_weights_and_boosts!(lp, ama_params, num_samples=num_samples, num_perturb=num_perturb, α=0.0, lr=lr, noise_magnitude=noise_magnitude, num_iters=num_training_iters)
    else
        (ama_params, vals) = optimize_boosts!(lp, ama_params, num_samples=num_samples, num_perturb=num_perturb, α=0.0, lr=lr, noise_magnitude=noise_magnitude, num_iters=num_training_iters)
    end
    end_time = Dates.now()
    @show vals

    vcg_revenue, vcg_std = expectedrevenue(lp, vcg_ama_params, num_samples=TEST_SAMPLES, α=0.00, require_optimal=true) # VCG revenue
    vcg_performance, vcg_performance_std = expectedperformance(lp, vcg_ama_params, num_samples=TEST_SAMPLES, α=0.00, require_optimal=true) # VCG revenue
    @show vcg_revenue
    @show vcg_std
    @show boosts
  
    ama_revenue, ama_std = expectedrevenue(lp, ama_params, num_samples=TEST_SAMPLES, α=0.00, require_optimal=true) # optimized revenue
    ama_performance, ama_performance_std = expectedperformance(lp, ama_params, num_samples=TEST_SAMPLES, α=0.0, require_optimal=true) # optimized revenue
    @show ama_revenue
    @show ama_std

    println("*****************")
    (
        (method="zeroorder",
        mdp=string(mdptype),
        num_agents=num_agents,
            num_items=num_items,
            seed=seed,
            num_samples=num_samples,
            test_samples=TEST_SAMPLES,
            num_perturb=num_perturb,
            lr=lr,
            noise_magnitude=noise_magnitude,
            num_training_iters=num_training_iters,
            dist_type=dist_type,
            vcg_performance=vcg_performance,
            vcg_performance_std=vcg_performance_std,
            ama_performance=ama_performance,
            ama_performance_std=ama_performance_std,
            vcg_revenue=vcg_revenue,
            vcg_std=vcg_std,
            ama_revenue=ama_revenue,
            ama_std=ama_std,
            runtime=(end_time - start_time)),
        (ama=ama_params, vals=vals)
    )
end

#beware for the infiinite MDP we have now added gamma because it has to be below 1
@main function main(
    num_agents::Int64,
    num_items::Int64,
    num_samples::Int64,
    num_perturb::Int64,
    num_training_iters::Int64,
    start_seed::Int64,
    num_trials::Int64,
    mdptype_str::String="auction",
    lr::Float64=0.01,
    noise_magnitude::Float64=0.01,
    dist_type::String="uniform",
    γ::Float64=1.0)

    @show num_agents
    @show num_items
    @show num_samples
    @show num_perturb
    @show num_training_iters
    @show start_seed
    @show num_trials
    @show mdptype_str
    @show lr
    @show noise_magnitude
    @show dist_type
    @show γ
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
    csv_filename = "results_zeroorder_" * mdptype_str * 
                "_agents_" * string(num_agents) *
                "_items_" * string(num_items) *
                "_samples_" * string(num_samples) *
                "_perturb_" * string(num_perturb) *
                "_training_iters_" * string(num_training_iters) *
                "_seed_" * string(start_seed) *
                "_trials_" * string(num_trials) *
                "_lr_" * string(lr) *
                "_noise_" * string(noise_magnitude) *
                "_dist_" * dist_type * ".csv"

    for trial = 1:num_trials
        @show trial, mdptype
        start_seed += 1
        boostvals_filename = "boostvals_zeroorder_" * mdptype_str *
                         "_agents_" * string(num_agents) *
                         "_items_" * string(num_items) *
                         "_samples_" * string(num_samples) *
                         "_perturb_" * string(num_perturb) *
                         "_training_iters_" * string(num_training_iters) *
                         "_seed_" * string(start_seed) *
                         "_trials_" * string(num_trials) *
                         "_trialnum_" * string(trial) * # Include the trial number
                         "_lr_" * string(lr) *
                         "_noise_" * string(noise_magnitude) *
                         "_dist_" * dist_type * ".jld2"
        if mdptype_str == "grid"
            use_weights = false
        else
            use_weights = dist_type == "asymmetric"
        end
        results, boosts_and_vals = runtrial(num_agents, num_items, num_samples, num_perturb, num_training_iters, start_seed, mdptype, lr, noise_magnitude, dist_type, optimize_weights=use_weights,γ=γ)
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
