importall POMDPs
using POMDPToolbox
using QMDP
using ParticleFilters
using BasicPOMCP

struct State
    # Latent variables
    desired_autonomy::Bool

    # Observable variables
    performance::Bool
    given_autonomy::Bool

    # last engagement, to be used for reward
    engagement::Bool
end

# struct Act 
#     give_autonomy::Bool
# end

struct Obs
    performance::Bool
    given_autonomy::Bool

    # Using duration (1 = engaged/'just right', 0 = too long / too short on task) 
    #   as a proxy for engagement
    duration::Bool

end

# Connect observation space to state space for observable variables
Obs(s::State) = Obs(s.performance, s.given_autonomy, s.engagement)

struct MOMDP <: POMDP{State, Symbol, Obs} #TODO mutable struct - ideally make p_ability change over time
    # CPT: P(u' | u, p, gu)
    p_autonomy_when_desired_good_given::Float64
    p_autonomy_when_desired_good_not_given::Float64
    p_autonomy_when_desired_bad_given::Float64
    p_autonomy_when_desired_bad_not_given::Float64
    p_autonomy_when_not_desired_good_given::Float64
    p_autonomy_when_not_desired_good_not_given::Float64
    p_autonomy_when_not_desired_bad_given::Float64
    p_autonomy_when_not_desired_bad_not_given::Float64

    # CPT: P(i' | u', gu)
    p_engaged_when_desired_given::Float64
    p_engaged_when_desired_not_given::Float64
    p_engaged_when_not_desired_given::Float64
    p_engaged_when_not_desired_not_given::Float64

    # For now, ability is a probabilistic constant for a student 
    #   that determines performance independent of attempt
    p_ability::Float64

    # Reward for being engaged ("just right", vs. took too long or too short ;
    #   using duration as a proxy for engagement)
    r_engagement::Float64

    discount::Float64
end

# Transition values from CPTs for default constructor
MOMDP() = MOMDP(0.9, 0.9, 0.3, 0.8, 0.8, 0.1, 0.01, 0.2,
                0.9, 0.3, 0.2, 0.9,
                0.5, # TODO: draw from distribution (first pass: tune manually to see diffs)
                1.0,
                0.95
                )

discount(m::MOMDP) = m.discount

const num_states = 2*2*2*2
const num_actions = 2
const num_observations = 2*2*2
n_states(::MOMDP) = num_states
n_actions(::MOMDP) = num_actions
n_observations(::MOMDP) = num_observations

# States of MOMDP
const all_states = [State(desired_autonomy, performance, given_autonomy, engagement) for engagement = 0:1, performance = 0:1, given_autonomy = 0:1, desired_autonomy = 0:1]
states(m::MOMDP) = all_states

""" Explicitly, the indices are (+1 on all these bc arrays in Julia are indexed at 1):
0 - 000 1 - 001 2 - 010 3 - 011
4 - 100 5 - 101 6 - 110 7 - 111
"""
function state_index(s::State)  
    # TODO: use sub2ind for efficiency
    return convert(Int64, s.desired_autonomy * 8 + s.performance * 4 + s.given_autonomy * 2 + s.engagement * 1 + 1)
end

state_index(m::MOMDP, s::State) = state_index(s)

# Actions of MOMDP
# const all_actions = [Act(give_autonomy) for give_autonomy = 0:1]
# actions(m::MOMDP) = all_actions

actions(m::MOMDP) = [:give_autonomy, :revoke_autonomy]

function action_index(m::MOMDP, a::Symbol)
    # TODO: use sub2ind for efficiency
    if a == :give_autonomy
        return 1
    elseif a == :revoke_autonomy
        return 2
    end
    error("invalid MOMDP action: $a")
end

const all_observations = [Obs(performance, given_autonomy, duration) for performance = 0:1, given_autonomy = 0:1, duration = 0:1]
observations(m::MOMDP) = all_observations

# Observation is certain
function observation(m::MOMDP, s::State)
    return SparseCat([Obs(s)], [1.0])
end

# Transition function P(s' | s, a)
function transition(m::MOMDP, s::State, a::Symbol, rng::AbstractRNG=MersenneTwister(1))
    sp_desired_autonomy = true
    sp_engagement = true
    sp_performance = rand(rng) < m.p_ability ? true : false

    # Next latent state of desired autonomy P(u' | u, p, gu)
    # If user wants autonomy
    if s.desired_autonomy
        # Does well
        if s.performance
            # And we give them autonomy
            if a == :give_autonomy
                # Then the prob for next desired_autonomy, and the given autonomy, updated in the state
                p_sp_desired_autonomy = m.p_autonomy_when_desired_good_given
                sp_given_autonomy = true
            else
                p_sp_desired_autonomy = m.p_autonomy_when_desired_good_not_given
                sp_given_autonomy = false
            end
        else
            if a == :give_autonomy
                p_sp_desired_autonomy = m.p_autonomy_when_desired_bad_given
                sp_given_autonomy = true
            else
                p_sp_desired_autonomy = m.p_autonomy_when_desired_bad_not_given
                sp_given_autonomy = false
            end
        end
    else
        if s.performance
            if a == :give_autonomy
                p_sp_desired_autonomy = m.p_autonomy_when_not_desired_good_given
                sp_given_autonomy = true
            else
                p_sp_desired_autonomy = m.p_autonomy_when_not_desired_good_not_given
                sp_given_autonomy = false
            end
        else
            if a == :give_autonomy
                p_sp_desired_autonomy = m.p_autonomy_when_not_desired_bad_given
                sp_given_autonomy = true
            else
                p_sp_desired_autonomy = m.p_autonomy_when_not_desired_bad_not_given
                sp_given_autonomy = false
            end
        end
    end

    # Next engagement level P(i' | u', gu)
    if sp_given_autonomy
        p_sp_engagement_desired = m.p_engaged_when_desired_given
        p_sp_engagement_not_desired = m.p_engaged_when_not_desired_given
    else
        p_sp_engagement_desired = m.p_engaged_when_desired_not_given
        p_sp_engagement_not_desired = m.p_engaged_when_not_desired_not_given
    end

    # Let's say performance is a general ability that's constant throughout the curriculum for now
    p_sp_performance = m.p_ability

    sps = State[]
    probs = Float64[]
    push!(sps, State(sp_desired_autonomy, sp_performance, sp_given_autonomy, sp_engagement))
    push!(probs, p_sp_desired_autonomy * p_sp_engagement_desired * p_sp_performance)
    push!(sps, State(!sp_desired_autonomy, sp_performance, sp_given_autonomy, sp_engagement))
    push!(probs, (1.0 - p_sp_desired_autonomy) * p_sp_engagement_not_desired * p_sp_performance)
    push!(sps, State(sp_desired_autonomy, sp_performance, sp_given_autonomy, !sp_engagement))
    push!(probs, p_sp_desired_autonomy * (1.0 - p_sp_engagement_desired) * p_sp_performance)
    push!(sps, State(!sp_desired_autonomy, sp_performance, sp_given_autonomy, !sp_engagement))
    push!(probs, (1.0 - p_sp_desired_autonomy) * (1.0 - p_sp_engagement_not_desired) * p_sp_performance)

    push!(sps, State(sp_desired_autonomy, !sp_performance, sp_given_autonomy, sp_engagement))
    push!(probs, p_sp_desired_autonomy * p_sp_engagement_desired * (1.0 - p_sp_performance))
    push!(sps, State(!sp_desired_autonomy, !sp_performance, sp_given_autonomy, sp_engagement))
    push!(probs, (1.0 - p_sp_desired_autonomy) * p_sp_engagement_not_desired * (1.0 - p_sp_performance))
    push!(sps, State(sp_desired_autonomy, !sp_performance, sp_given_autonomy, !sp_engagement))
    push!(probs, p_sp_desired_autonomy * (1.0 - p_sp_engagement_desired) * (1.0 - p_sp_performance))
    push!(sps, State(!sp_desired_autonomy, !sp_performance, sp_given_autonomy, !sp_engagement))
    push!(probs, (1.0 - p_sp_desired_autonomy) * (1.0 - p_sp_engagement_not_desired) * (1.0 - p_sp_performance))
    
    # Debugging
    # print("\n######\n")
    # print(s, " desired_autonomy, performance, given_autonomy, engagement\n", a, "\n")
    # print(sps, "\n")
    # print(probs, "\n")
    # print("\n######\n")
    return SparseCat(sps, probs)
end

# Rewarded for being engaged
function POMDPs.reward(m::MOMDP, s::State, a::Symbol)
    return s.engagement ? m.r_engagement : 0.0 #TODO: try -1.0 here
end

# initial_state_distribution(m::MOMDP) = SparseCat(states(m), ones(num_states) / num_states)
p_initially_motivated = 0.5 # 0.5 is uniform prior
init_state_dist = SparseCat([State(true, false, false, false), State(false, false, false, false)], [p_initially_motivated, 1.0-p_initially_motivated])
initial_state_distribution(m::MOMDP) = init_state_dist

# Solver

momdp = MOMDP()

# QMDP 
solver = QMDPSolver(max_iterations=100, tolerance=1e-3) 
policy = solve(solver, momdp, verbose=true)
# print(policy)

filter = SIRParticleFilter(momdp, 10000)

init_dist = initial_state_distribution(momdp)
init_belief = initialize_belief(filter, init_dist)

# function unroll_sparse_cat(d::POMDPToolbox.SparseCat{Array{State,1}})
#     state_names = d.vals
#     state_probs = d.probs

#     dp = [0.0 for i = 1:num_states]
#     for (ni, n) in enumerate(state_names)
#         si = state_index(n)
#         dp[si] = state_probs[ni]
#     end
#     return dp
# end

function input(ask::String="performance")::Bool
    if ask == "performance"
        prompt = "performed well? (y/n) "
    else
        prompt = "engaged? (y/n) "
    end
    print(prompt)
    user_input = chomp(readline())
    if user_input == "n"
        return false
    else
        return true
    end
end

function unroll_particles(particles::ParticleFilters.ParticleCollection{State})
    d = [0.0 for i = 1:num_states]
    d = []
    for i = 1:num_states
        push!(d, pdf(particles, all_states[i]))
    end
    return d
end

function generate_next_action(particle_belief::Any=init_belief, iteration::Int64=1)
    println("Step: ", iteration)
    belief = unroll_particles(particle_belief)
    println("Belief: ", belief)
    _, action_idx = findmax(belief .* policy.alphas)

    if action_idx <= size(belief)[1]
        action = :give_autonomy
        action_val = true
    else
        action = :revoke_autonomy
        action_val = false
    end
    println("Next action is: ", action)

    # Input user's performance and engagement
    inputed_performance = input("performance")
    inputed_engagement = input("engagement")

    o = Obs(inputed_performance, action_val, inputed_engagement)
    next_belief = update(filter, particle_belief, action, o)
    return generate_next_action(next_belief, iteration + 1)
end

generate_next_action()

#=
# Visualize D3tree setup
text = ["revoke"]
tree = [[2,3,4,5]]
node_idx = 6
for node in tree
    # depth d requires # nodes = 2*(4^0+4^1+4^2+4^3+...4^d) (mult by 2 b/c it's num expansion nodes + num extra action nodes)
    # d=30 is 3.0744573e+18 (copy paste below for chrome calculator evaluation)
    # 2*(4^0+4^1+4^2+4^4+4^5+4^6+4^7+4^8+4^9+4^10+4^11+4^12+4^13+4^14+4^15+4^16+4^17+4^18+4^19+4^20+4^21+4^22+4^23+4^24+4^25+4^26+4^27+4^28+4^29+4^30)=
    if node_idx < 2796074
        if size(node, 1) == 1
            # add a 4 child next
            push!(tree, [node_idx, node_idx+1, node_idx+2, node_idx+3])
            # increment node_idx by 4
            node_idx += 4

            # add text
            push!(text, "action")
        elseif size(node, 1) == 4
            # add a 1 child next
            push!(tree, [node_idx])
            push!(tree, [node_idx+1])
            push!(tree, [node_idx+2])
            push!(tree, [node_idx+3])
            # increment node_idx by 1
            node_idx += 4

            # add text
            push!(text, "good eng")
            push!(text, "good dis")
            push!(text, "bad eng")
            push!(text, "bad dis")
        elseif size(node, 1) < 1
            # terminate this process when reached leaf nodes
            print("terminate", size(tree))
            break
        end
    end
end

for seed = 1:10
    hist = HistoryRecorder(max_steps=10, rng=MersenneTwister(seed), show_progress=true)
    hist = simulate(hist, momdp, policy, filter, init_dist)
    tree_node_idx = 1
    for (a, o, bp) in eachstep(hist, "(a, o, bp)")
        # If a is already in there, don't update, else update "action" to be the desired action
        if text[tree_node_idx] == "action"
            a = replace(string(a), "_autonomy", "")
            text[tree_node_idx] = a
        end

        if o.performance
            if o.duration
                obs = "good eng"
                obs_val = 1
            else
                obs = "good dis"
                obs_val = 2
            end
        else
            if o.duration
                obs = "bad eng"
                obs_val = 3
            else
                obs = "bad dis"
                obs_val = 4
            end
        end

        # Update index for observation node
        tree_node_idx = tree[tree_node_idx][obs_val]

        # If belief text is already there ([] in string), don't update, else update the o node with this belief
        belief = QMDP.belief_vector(policy, bp)
        belief = convert(Array, belief)
        filter!(e->e!=0.0, belief)
        if contains(text[tree_node_idx], "[")
            # do nothing
        else
            text[tree_node_idx] = obs * "\n" * string(belief)
        end
        # Update index for next action node
        tree_node_idx = tree[tree_node_idx][1]
    end
end

# Modified inchrome() to work on Mac
function chrome_display(t::D3Tree)
    fname = joinpath(mktempdir(), "tree.html")
    open(fname, "w") do f
        show(f, MIME("text/html"), t)
    end
    run(`open -a "Google Chrome" $fname`)
end

using D3Trees
dtree = D3Tree(tree, text=text, init_expand=2)
chrome_display(dtree)
=#


