using Test
include("grid_search.jl")
#include("gridworld.jl")



grid_size = 3
n_agents = 2
γ = 0.9
dist_type = "onecell"
grid = GridMDP(n_agents, grid_size, γ, dist_type)
@test grid.n_items == grid_size
@test grid.n_agents == n_agents
@test grid.γ == γ
@test grid.dist_type == dist_type
@test grid.state_list == vec([(1, 1), (2,1), (3,1), (1,2), (2,2), (3,2), (1,3), (2,3), (3,3)])
@test grid.action_list == ["up", "down", "left", "right"]
@test startstate(grid) in grid.state_list
@test transition_probability(grid, (1, 1), "up", (1, 2)) == 0
@test transition_probability(grid, (1, 1), "up", (1, 3)) == 1
@test transition_probability(grid, (1, 1), "down", (1, 1)) == 0
@test transition_probability(grid, (1, 1), "down", (1, 2)) == 1
@test transition_probability(grid, (1, 1), "left", (1, 1)) == 0
@test transition_probability(grid, (1, 1), "left", (3, 1)) == 1
@test transition_probability(grid, (1, 1), "right", (1, 1)) == 0
@test transition_probability(grid, (1, 1), "right", (2, 1)) == 1
@test transition_probability(grid, (3,2), "up", (3, 2)) == 0
@test transition_probability(grid, (3,2), "up", (3, 1)) == 1
@test transition_probability(grid, (3,2), "down", (3, 2)) == 0
@test transition_probability(grid, (3,2), "down", (3, 3)) == 1
@test transition_probability(grid, (2,3), "down", (3, 2)) == 0
@test transition_probability(grid, (2,3), "down", (2, 1)) == 1

num_samples =2
types = sampletypes(grid, num_samples)
# types should be a matrix of size 1 x n_agents x grid_size x grid_size
@test size(types) == (num_samples,)
# All entries should be between 0 and 1
@test all([all(0 .<= types[i]) for i in 1:num_samples]) && all([all(types[i] .<= 1) for i in 1:num_samples])
# For each sample (first dimension) and each agent (2nd dimension), 
# all cells should have value 0 except for one, which can have any value between 0 and 1

function has_only_one_non_zero(mat::AbstractMatrix)
    return sum(mat .!= 0) == 1
end

results = Array{Bool}(undef, 1, n_agents)

# Loop over all combinations of the first two coordinates
for i in 1:1  # since the first dimension has size 1
    for j in 1:n_agents
        results[i, j] = has_only_one_non_zero(types[i][ j, :, :])
    end
end
@test all(results)

# Do a grid_search

function get_actions(matrix,action_list)
    actions_grid = fill("", 3, 3)
    for (i, row) in enumerate(eachrow(matrix))
        max_index = argmax(row)
        actions_grid[div(i-1,3)+1, rem(i-1,3)+1] = action_list[max_index]
    end
    return actions_grid
end

num_agents = 2
grid_size = 3
num_samples = 100
start_seed = 1

dist_type = "onecell"
use_weights = false
num_samples_evaluation = 1000


mdp = GridMDP(num_agents, grid_size, 0.9, dist_type)
α = 0.00
@show length(mdp.state_list)
lp = MDPLinearProgram(mdp)
weights = ones(mdp.n_agents)

vcg_weights = ones(lp.mdp.n_agents)
ama = AMAParams(vcg_weights, zeros(size(lp.x, 1), size(lp.x, 2)))
types = sampletypes(lp.mdp, num_samples)
types =  Array{Float64, 3}(undef, 2, 3, 3)
types[1,:,:] = [0.0 0.0 0.7458095117719394; 0.0 0.0 0.0; 0.0 0.0 0.0]
types[2,:,:] = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.5929855446281721 0.0 0.0]

evalauction!(lp, types, ama, α)

xx = value.(lp.x)
action_list = mdp.action_list
actions_grid = get_actions(xx,action_list)

println("actions_grid: ", actions_grid)
# print out types of agent 1 and 2 as transposed matrix
type1 = transpose(types[1,:,:])
for i in 1:3
    println(type1[i,:])
end
type2 = transpose(types[2,:,:])
for i in 1:3
    println(type2[i,:])
end

counterfactualtypes1 = counterfactualtype(lp.mdp, types, 1)
evalauction!(lp, counterfactualtypes1, ama, α)
xx = value.(lp.x)
action_list = mdp.action_list
actions_grid = get_actions(xx,action_list)

counterfactualtypes2 = counterfactualtype(lp.mdp, types, 2)
evalauction!(lp, counterfactualtypes2, ama, α)
xx = value.(lp.x)
action_list = mdp.action_list
actions_grid = get_actions(xx,action_list)

weird_boosts =[10.000000000000004 10.000000000000004 10.000000000000004 10.000000000000004; -10.000000000000002 10.000000000000004 -10.000000000000002 -10.000000000000002; -10.000000000000002 10.000000000000004 -10.000000000000002 10.000000000000004; -10.000000000000002 -10.000000000000002 10.000000000000004 -10.000000000000002; 10.000000000000004 -10.000000000000002 -10.000000000000002 -10.000000000000002; 10.000000000000004 10.000000000000004 -10.000000000000002 -10.000000000000002; -10.000000000000002 -10.000000000000002 -10.000000000000002 10.000000000000004; 10.000000000000004 10.000000000000004 10.000000000000004 10.000000000000004; 10.000000000000004 -10.000000000000002 -10.000000000000002 10.000000000000004]
grad_boosts = [0.394327077345363 2.073477387973632 0.6972019717628715 0.47869504975816995; 0.38126190878742083 -0.37698042631734613 -1.7801143455651192 0.12152929978866084; 0.945473962934343 -0.6363301516015156 0.5622178446553092 1.3316770647450817; 0.24260489367127547 -0.31712929210617785 -0.14620178894465152 -0.06824200788734061; 0.0594484025341534 -0.2762304399538271 1.300345558690015 1.1377746261046764; 0.7511238518149796 0.13638676109435294 1.0830445963994437 0.5300729817598604; 1.4994417516772343 1.0624539524016585 1.3153145574436824 0.6519812358057866; 1.6644801150479938 0.35424059188911444 1.0822096032539215 -0.4841096200257151; 0.4549644738616962 -0.18685302653517072 0.16461697177383067 0.5959386021202666]
lp_boosts = [0.6511323264773393 0.2956472959339353 0.6043693524108597 0.04010791590594832; 0.8882016691730727 0.11287429786108985 0.13541728140491902 0.09941896907968324; 0.4721220850314363 0.8246210721185675 0.6395019605731366 0.7529989536661412; 0.7873776373044536 0.372028510873892 0.7654988995217877 0.8927412060070063; 0.6142597901379169 0.4716930192148046 0.5212462904541619 0.8656756795324246; 0.28975817870649084 0.3266899120211333 0.07835642022306738 0.641483556251341; 0.2244102907505516 0.576775362232549 0.2203801292001616 0.3875489913733064; 0.05455757724852569 0.35998854957973414 0.2431046989435406 0.31336255917676314; 0.8376062769968048 0.4877523928427872 0.2226052075326758 0.09750669496315781]
cluster_4_2_4_boosts = [0.1593444233678684 0.965451319945199 0.2984625110476483 0.35102535941091273; 0.6791651236762288 0.6669088624324288 0.019529711492724828 0.5615445165866607; 0.8842901891800306 0.7305694859059575 0.04459410983626188 0.613550286598956; 0.5686048596049315 0.9463117137452277 0.49163337154710723 0.5538907456734955; 0.9082481872598627 0.4747387754041348 0.5037213314147203 0.30251067191117886; 0.6979967616501888 0.5970312339271017 0.31768485610647457 0.3694219137295178; 0.6612490227387319 0.663798445304066 0.33852635439137824 0.17740307853493978; 0.3125599346790135 0.06775527330151404 0.4067669552179386 0.14117187506575488; 0.9268999339498973 0.7202455998636998 0.30306066723369973 0.1843976822865092; 0.6676165160467006 0.8263106931741484 0.033648233987600674 0.4601903775850323; 0.06787907340248217 0.2402564312334063 0.08233763417665885 0.3932339336409479; 0.8141100416067525 0.04268230166282785 0.9394476343355117 0.8101621891825306; 0.6459147766935146 0.4550039543762289 0.5366854636429068 0.20717483976881018; 0.5751878124588952 0.4299606648983666 0.8089547672260117 0.36581980855885715; 0.16293456484455549 0.5564259407386011 0.020884475747622543 0.7254813231021268; 0.5605775418821072 0.8351499228862341 0.5465981483901501 0.27390111652002497]
cluster_3_2_3_unif_boosts = [0.4391466612502235 0.5248854715003474 0.7373816599738718 1.282296656739283; 0.5174231557009796 0.12168309755131336 1.406949387058296 0.6232633128787065; 0.3007756601018027 0.2199931495107866 0.6847999130642412 0.7255810875272314; 0.32349329760764756 0.3532720468073499 0.5783696857434467 0.3736103901863781; 0.0774306456169359 0.45256263578539646 0.6605691437464603 0.215278556014773; 0.5207059295522729 0.6813872262019185 0.7101621594719386 0.487973349572429; 0.573572763026199 0.3221116308891536 0.27157916882765054 0.030123806390907046; 0.6674480863074002 0.7821877706335957 0.8280461451079743 0.17378103813342127; 0.5873654427094941 0.7044171086872276 0.33489948112688495 0.6756244062658022]

function get_all_actions(matrix, action_list,size)
    actions_grid = fill("", size, size)
    for (i, row) in enumerate(eachrow(matrix))
        max_value = maximum(row)
        max_indices = findall(x -> x == max_value, row)
        actions = join(action_list[max_indices], "/")
        actions_grid[div(i-1, size)+1, rem(i-1, size)+1] = actions
    end
    return actions_grid
end
get_all_actions(cluster_4_2_4_boosts,action_list,4)
get_all_actions(cluster_3_2_3_unif_boosts,action_list,3)
actions_grid = get_all_actions(lp_boosts,action_list)

ama = AMAParams(vcg_weights, weird_boosts ./10)

performance = expectedperformance(lp, ama, num_samples = 10000,α= 0.00, require_optimal=true)
println("VCG performance: ", performance)






