importall POMDPs
using POMDPToolbox
using QMDP
using ParticleFilters
using BasicPOMCP

# state: State=struct, action: Act=struct, obs: Obs=struct
struct State
    # knowledge::[Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool] # Bit array repr. each node in graph

    # Latent variables
    desired_autonomy::Bool
    # desired_difficulty::Bool

    # Observable variables
    performance::Bool #TODO: Float64 or Int64
    # exercised_autonomoy::Bool
    # exercised_difficulty::Bool
    given_autonomy::Bool
    # given_difficulty::Bool

    # last engagement, to be used for reward
    engagement::Bool #TODO: Float64 or Int64
end

struct Act 
    give_autonomy::Bool
    # give_difficulty::Bool
end

struct Obs
    # Using duration (1 = engaged/'just right', 0 = too long / too short on task) 
    #   as a proxy for engagement
    engagement::Bool #TODO: Float64 or Int64

    performance::Bool #TODO: Float64 or Int64
    # exercised_autonomoy::Bool
    # exercised_difficulty::Bool
    given_autonomy::Bool
    # given_difficulty::Bool 
end

# Connect observation space to state space for observable variables
# Obs(s::State, d::Bool) = Obs(d, s.performance, s.given_autonomy)
Obs(s::State) = Obs(s.engagement, s.performance, s.given_autonomy)

struct MOMDP <: POMDP{State, Act, Obs} #TODO mutable struct - ideally make p_ability change over time
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

    # For now, ability is a stochastic constant for a student 
    #   that determines performance independent of attempt
    p_ability::Float64

    # Reward for being engaged ("just right", vs. took too long or too short ;
    #   using duration as a proxy for engagement)
    r_engagement::Float64

    discount::Float64
end

# Transition values from CPTs for default constructor
MOMDP() = MOMDP(0.99, 0.9, 0.3, 0.8, 0.8, 0.1, 0.01, 0.2,
                0.9, 0.7, 0.2, 0.8,
                0.99, # TODO: draw from distribution (first pass: tune manually to see diffs)
                1.0,
                1.0
                )

discount(m::MOMDP) = m.discount

function generate_s(m::MOMDP, s::State, a::Act, rng::AbstractRNG)
    # If user wants autonomy
    if s.desired_autonomy
        # Does well
        if s.performance
            # And we give them autonomy
            if a.give_autonomy
                # Then the prob for next desired_autonomy, and the given autonomy, updated in the state
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_performed_given ? true : false
                sp_given_autonomy = true
            else
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_performed_not_given ? true : false
                sp_given_autonomy = false
            end
        else
            if a.give_autonomy
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_bad_given ? true : false
                sp_given_autonomy = true
            else
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_bad_given ? true : false
                sp_given_autonomy = false
            end
        end
    else
        if s.performance
            if a.give_autonomy
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_performed_given ? true : false
                sp_given_autonomy = true
            else
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_performed_not_given ? true : false
                sp_given_autonomy = false
            end
        else
            if a.give_autonomy
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_bad_given ? true : false
                sp_given_autonomy = true
            else
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_bad_given ? true : false
                sp_given_autonomy = false
            end
        end
    end
    sp = State(sp_desired_autonomy, s.performance, sp_given_autonomy, s.engagement)
    return sp
end

function generate_o(m::MOMDP, s::State, a::Act, sp::State, rng::AbstractRNG)
    # If the user wants autonomy in this next state
    if sp.desired_autonomy
        # And was given autonomy in this next state
        if sp.given_autonomy
            engagement = rand(rng) < m.p_engaged_when_desired_given ? true : false
        else
            engagement = rand(rng) < m.p_engaged_when_desired_not_given ? true : false
        end
    else
        if sp.given_autonomy
            engagement = rand(rng) < m.p_engaged_when_not_desired_given ? true : false
        else
            engagement = rand(rng) < m.p_engaged_when_desired_not_given ? true : false
        end
    end
    # Let's say performance is a general ability that's constant throughout the curriculum for now
    p = rand(rng) < m.p_ability ? true : false
    return Obs(engagement, p, sp.given_autonomy)
end

POMDPs.discount(m::MOMDP) = m.discount
num_states = 2*2*2*2
num_actions = 2
num_observations = 2*2*2
POMDPs.n_states(::MOMDP) = num_states
POMDPs.n_actions(::MOMDP) = num_actions
POMDPs.n_observations(::MOMDP) = num_observations

# States of MOMDP
function POMDPs.states(m::MOMDP)
    s = State[]
    for desired_autonomy = 0:1, performance = 0:1, given_autonomy = 0:1, engagement = 0:1
        push!(s, State(desired_autonomy, performance, given_autonomy, engagement))
    end
    return s
end

""" Explicitly, the indices are (+1 on all these bc arrays in Julia are indexed at 1):
0 - 000 1 - 001 2 - 010 3 - 011
4 - 100 5 - 101 6 - 110 7 - 111
"""
function state_index(s::State)   
    return convert(Int64, s.desired_autonomy * 8 + s.performance * 4 + s.given_autonomy * 2 + s.engagement * 1 + 1)
end

state_index(m::MOMDP, s::State) = state_index(s)

# Actions of MOMDP
function POMDPs.actions(m::MOMDP)
    a = Act[]
    for give_autonomy = 0:1
        push!(a, Act(give_autonomy))
    end
    return a
end

function action_index(m::MOMDP, a::Act)
    if a.give_autonomy == 0
        # NB: Arrays in Julia are indexed at 1, not 0
        return 1
    else
        return 2
    end
end

# Observations of MOMDP
function POMDPs.observations(m::MOMDP)
    o = Obs[]
    for engagement = 0:1, performance = 0:1, given_autonomy = 0:1
        push!(s, Obs(engagement, performance, given_autonomy))
    end
    return o
end

# Linked observation vars with state vars
POMDPs.observations(m::MOMDP, s::State) = observations(m)

# Iterator for going over all states
struct StateSpace
    min_index::Int64
    max_index::Int64
end
POMDPs.iterator(s::State) = StateSpace(1, num_states)

# Transition distribution
struct Distribution
    p::Float64
    it::StateSpace
end

# Default constructor for transition is 50/50
Distribution() = Distribution(0.5, StateSpace(1, num_states))

POMDPs.iterator(d::Distribution) = d.it

function POMDPs.pdf(d::Distribution, s::State)
    s_index = state_index(s)
    not_s = State(!s.desired_autonomy, s.performance, s.given_autonomy, s.engagement)
    not_s_index = state_index(not_s)
    return SparseCat([s_index, not_s_index], [d.p, 1.0-d.p])
end

# Observation is certain
function POMDPs.observation(m::MOMDP, s::Obs)
    return SparseCat([o], [1.0])
end

# Transition function P(s' | s, a)
function POMDPs.transition(m::MOMDP, s::State, a::Act)
    d = Distribution()
    if a.give_autonomy == true
        sp = State(s.desired_autonomy, s.performance, true, s.engagement)
    else
        sp = State(s.desired_autonomy, s.performance, false, s.engagement)
    end
    return pdf(d, sp)
    # rng = MersenneTwister(0)
    # sp = generate_s(m, s, a, rng)
    # op = generate_o(m, s, a, sp, rng)
end


# Rewarded for being engaged
function POMDPs.reward(m::MOMDP, s::State, a::Act)
    return s.engagement ? m.r_engagement : 0.0
    # return o.engagement ? p.r_engagement : 0.0
end

# initial_state_distribution(m::MOMDP) = [[true, false], false, false]
POMDPs.initial_state_distribution(m::MOMDP) = Distribution()

POMDPs.rand(rng::AbstractRNG, d::Distribution) = rand(rng) <= d.p;


# Solver

momdp = MOMDP()

# QMDP 
# printed (10 max_iter): QMDP.QMDPPolicy{MOMDP,Act}([0.0 0.0; 18.375 18.875; 0.0 0.0; 19.375 18.875; 0.0 0.0; 18.375 18.875; 0.0 0.0; 19.375 18.875; 0.0 0.0; 19.375 19.875; 0.0 0.0; 20.375 19.875; 0.0 0.0; 19.375 19.875; 0.0 0.0; 20.375 19.875], Act[Act(false), Act(true)], MOMDP(0.99, 0.9, 0.3, 0.8, 0.8, 0.1, 0.01, 0.2, 0.9, 0.7, 0.2, 0.8, 0.5, 1.0, 1.0))
# printed (100 max_iter): QMDP.QMDPPolicy{MOMDP,Act}([0.0 0.0; 198.375 198.875; 0.0 0.0; 199.375 198.875; 0.0 0.0; 198.375 198.875; 0.0 0.0; 199.375 198.875; 0.0 0.0; 199.375 199.875; 0.0 0.0; 200.375 199.875; 0.0 0.0; 199.375 199.875; 0.0 0.0; 200.375 199.875], Act[Act(false), Act(true)], MOMDP(0.99, 0.9, 0.3, 0.8, 0.8, 0.1, 0.01, 0.2, 0.9, 0.7, 0.2, 0.8, 0.5, 1.0, 1.0))

# solver = QMDPSolver(max_iterations=100, tolerance=1e-3) 
# policy = solve(solver, momdp, verbose=true)
# print(policy)

# filter = SIRParticleFilter(momdp, 10000)

# init_dist = initial_state_distribution(momdp)

# up = updater(policy)

# hist = HistoryRecorder(max_steps=20, rng=MersenneTwister(1), show_progress=true)
# hist = simulate(hist, momdp, policy, up, init_dist, filter)

# # Print statements not working...
# print(hist)

# for (s, b, a, r, sp, op) in hist
#     println("s: $s, b: $(b.b), action: $a, obs: $op")
# end
# println("Total reward: $(discounted_reward(hist))")

# POMCPSolver

solver = POMCPSolver()
policy = solve(solver, momdp)
# print(policy)

for (s, a, o) in stepthrough(momdp, policy, "sao", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end







