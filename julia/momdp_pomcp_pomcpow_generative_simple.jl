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

# function generate_sr(m::MOMDP, s::State, a::Act, rng::AbstractRNG)
#     (sp, o) = generate_so(m, s, a, rng)
#     r = sp.engagement ? sp.r_engagement : 0.0
#     return (sp, r)
# end

# function generate_sor(m::MOMDP, s::State, a::Act, rng::AbstractRNG)
#     (sp, o) = generate_so(m, s, a, rng)
#     r = sp.engagement ? sp.r_engagement : 0.0
#     return (sp, o, r)
# end

POMDPs.discount(m::MOMDP) = m.discount

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

# Rewarded for being engaged
function POMDPs.reward(m::MOMDP, s::State, a::Act)
    return s.engagement ? m.r_engagement : 0.0
    # return o.engagement ? p.r_engagement : 0.0
end

# initial_state_distribution(m::MOMDP) = [[true, false], false, false]
POMDPs.initial_state_distribution(m::MOMDP) = Distribution()

POMDPs.rand(rng::AbstractRNG, d::Distribution) = rand(rng) <= d.p;


# POMCPSolver
using POMCPOW
using POMDPModels
using POMDPToolbox
using BasicPOMCP

momdp = MOMDP()

# solver = POMCPSolver()
solver = POMCPOWSolver(criterion=MaxUCB(20.0))
policy = solve(solver, momdp)
# print(policy)

for (s, a, o) in stepthrough(momdp, policy, "sao", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end







