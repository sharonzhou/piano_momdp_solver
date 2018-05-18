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
                0.01, # TODO: draw from distribution (first pass: tune manually to see diffs)
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
const all_states = [State(desired_autonomy, performance, given_autonomy, engagement) for desired_autonomy = 0:1, performance = 0:1, given_autonomy = 0:1, engagement = 0:1]
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
function transition(m::MOMDP, s::State, a::Symbol)
    sp_desired_autonomy = true
    sp_engagement = true
    sp_performance = true

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

initial_state_distribution(m::MOMDP) = SparseCat(states(m), ones(num_states) / num_states)
# initial_state_distribution(m::MOMDP) = SparseCat([State(false, true, false, true)], [1.0])

# Solver

momdp = MOMDP()

# QMDP 
solver = QMDPSolver(max_iterations=100, tolerance=1e-3) 
policy = solve(solver, momdp, verbose=true)
print(policy)

filter = SIRParticleFilter(momdp, 10000)

init_dist = initial_state_distribution(momdp)

hist = HistoryRecorder(max_steps=20, rng=MersenneTwister(1), show_progress=true)
hist = simulate(hist, momdp, policy, filter, init_dist)

children = Array[]
for (s, b, a, r, sp, op) in hist
    println("s(desired_autonomy, performance, given_autonomy, engagement): $s, action: $a, obs(performance, given_autonomy, duration): $op")
    # push!(children, a, op)
    # println(QMDP.value(policy, b))
    # println(QMDP.action(policy, b))
    # println(QMDP.belief_vector(policy, b))
    # println(QMDP.unnormalized_util(policy, b))
    # println(policy.action_map)
    # println("s: $s, b: $(b.b), action: $a, obs: $op")
end
println("Total reward: $(discounted_reward(hist))")

# print(children)
# print(typeof(children))
# tree = D3Trees(children)
# print(tree)

for s in states(momdp)
    # @show s
    @show action(policy, ParticleCollection([s]))
    @printf("State(desired_autonomy=%s, performance=%s, given_autonomy=%s, engagement=%s) = Action(give_autonomy=%s)\n", s.desired_autonomy, s.performance, s.given_autonomy, s.engagement, action(policy, ParticleCollection([s])))
end

println(QMDP.alphas(policy))
# for (action_true, action_false) in policy.alphas
#     println("$action_true")
# end
# using D3Trees

# D3Tree(policy, State(true, true, true, true), init_expand=2)

# POMCPSolver

# solver = POMCPSolver()
# policy = solve(solver, momdp)
# # print(policy)

# for (s, a, o) in stepthrough(momdp, policy, "sao", max_steps=10)
#     println("State was $s,")
#     println("action $a was taken,")
#     println("and observation $o was received.\n")
# end


# Generative model (cannot use for QMDP)
# function generate_s(m::MOMDP, s::State, a::Act, rng::AbstractRNG)
#     # If user wants autonomy
#     if s.desired_autonomy
#         # Does well
#         if s.performance
#             # And we give them autonomy
#             if a.give_autonomy
#                 # Then the prob for next desired_autonomy, and the given autonomy, updated in the state
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_good_given ? true : false
#                 sp_given_autonomy = true
#             else
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_good_not_given ? true : false
#                 sp_given_autonomy = false
#             end
#         else
#             if a.give_autonomy
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_bad_given ? true : false
#                 sp_given_autonomy = true
#             else
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_bad_given ? true : false
#                 sp_given_autonomy = false
#             end
#         end
#     else
#         if s.performance
#             if a.give_autonomy
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_good_given ? true : false
#                 sp_given_autonomy = true
#             else
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_good_not_given ? true : false
#                 sp_given_autonomy = false
#             end
#         else
#             if a.give_autonomy
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_bad_given ? true : false
#                 sp_given_autonomy = true
#             else
#                 sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_bad_given ? true : false
#                 sp_given_autonomy = false
#             end
#         end
#     end
#     sp = State(sp_desired_autonomy, s.performance, sp_given_autonomy, s.engagement)
#     return sp
# end

# function generate_o(m::MOMDP, s::State, a::Act, sp::State, rng::AbstractRNG)
#     # If the user wants autonomy in this next state
#     if sp.desired_autonomy
#         # And was given autonomy in this next state
#         if sp.given_autonomy
#             engagement = rand(rng) < m.p_engaged_when_desired_given ? true : false
#         else
#             engagement = rand(rng) < m.p_engaged_when_desired_not_given ? true : false
#         end
#     else
#         if sp.given_autonomy
#             engagement = rand(rng) < m.p_engaged_when_not_desired_given ? true : false
#         else
#             engagement = rand(rng) < m.p_engaged_when_desired_not_given ? true : false
#         end
#     end
#     # Let's say performance is a general ability that's constant throughout the curriculum for now
#     p = rand(rng) < m.p_ability ? true : false
#     return Obs(engagement, p, sp.given_autonomy)
# end







