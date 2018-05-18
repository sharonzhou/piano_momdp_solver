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
    exercised_autonomy::Bool
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
    exercised_autonomy::Bool
    # exercised_difficulty::Bool
    given_autonomy::Bool
    # given_difficulty::Bool 
end

# Connect observation space to state space for observable variables
# Obs(s::State, engagement::Bool) = Obs(engagement, s.performance, s.given_autonomy)
Obs(s::State) = Obs(s.engagement, s.performance, s.exercised_autonomy, s.given_autonomy)

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

    # CPT: P(i' | u', gu) -> TODO SIMPLE P(i' | u', eu')
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
                0.99, 0.3, 0.2, 0.9,
                0.99, # TODO: draw from distribution (first pass: tune manually to see diffs)
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
const all_actions = [Act(give_autonomy) for give_autonomy = 0:1]
actions(m::MOMDP) = all_actions

function action_index(m::MOMDP, a::Act)
    # TODO: use sub2ind for efficiency
    if a.give_autonomy == 0
        # NB: Arrays in Julia are indexed at 1, not 0
        return 1
    else
        return 2
    end
end

const all_observations = [Obs(engagement, performance, given_autonomy) for engagement = 0:1, performance = 0:1, given_autonomy = 0:1]
observations(m::MOMDP) = all_observations

# Observation is certain
function observation(m::MOMDP, s::State)
    return SparseCat([Obs(s)], [1.0])
end

# Transition function P(s' | s, a)
function transition(m::MOMDP, s::State, a::Act)
    rng = MersenneTwister(1)
    # Next latent state of desired autonomy P(u' | u, p, gu)
    # If user wants autonomy
    if s.desired_autonomy
        # Does well
        if s.performance
            # And we give them autonomy
            if a.give_autonomy
                # Then the prob for next desired_autonomy, and the given autonomy, updated in the state
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_good_given ? true : false
                sp_given_autonomy = true
            else
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_desired_good_not_given ? true : false
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
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_good_given ? true : false
                sp_given_autonomy = true
            else
                sp_desired_autonomy = rand(rng) < m.p_autonomy_when_not_desired_good_not_given ? true : false
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

    # Next engagement level P(i' | u', gu)
    # If the user wants autonomy in this next state
    if sp_desired_autonomy
        # And was given autonomy in this next state
        if sp_given_autonomy
            sp_engagement = rand(rng) < m.p_engaged_when_desired_given ? true : false
            not_sp_engagement = rand(rng) < m.p_engaged_when_not_desired_given ? true : false
        else
            sp_engagement = rand(rng) < m.p_engaged_when_desired_not_given ? true : false
            not_sp_engagement = rand(rng) < m.p_engaged_when_not_desired_not_given ? true : false
        end
    else
        if sp_given_autonomy
            sp_engagement = rand(rng) < m.p_engaged_when_not_desired_given ? true : false
            not_sp_engagement = rand(rng) < m.p_engaged_when_desired_given ? true : false
        else
            sp_engagement = rand(rng) < m.p_engaged_when_not_desired_not_given ? true : false
            not_sp_engagement = rand(rng) < m.p_engaged_when_desired_not_given ? true : false
        end
    end

    # Let's say performance is a general ability that's constant throughout the curriculum for now
    sp_performance = rand(rng) < m.p_ability ? true : false

    sp = State(sp_desired_autonomy, sp_performance, sp_given_autonomy, sp_engagement)
    not_sp = State(!sp.desired_autonomy, sp_performance, sp_given_autonomy, not_sp_engagement)
    
    # Probability of correct latent determination of state - for now, just fixing this to 1.0 since using probs above already to decide
    p = 1.0  #TODO: make not 1.0 so that not_sp has value here, else can now just do SparseCat([sp], [p])
    return SparseCat([sp, not_sp], [p, 1.0-p])
end

# Rewarded for being engaged
function POMDPs.reward(m::MOMDP, s::State, a::Act)
    return s.engagement ? m.r_engagement : 0.0 #TODO: try -1.0 here
end

# initial_state_distribution(m::MOMDP) = [[true, false], false, false]
initial_state_distribution(m::MOMDP) = SparseCat(states(m), ones(length(states(m))) / length(states(m)))

# QMDP Solver
momdp = MOMDP()
solver = QMDPSolver(max_iterations=100, tolerance=1e-3) 
policy = solve(solver, momdp, verbose=true)
print(policy)

filter = SIRParticleFilter(momdp, 10000)

init_dist = initial_state_distribution(momdp)

hist = HistoryRecorder(max_steps=20, rng=MersenneTwister(1), show_progress=true)
hist = simulate(hist, momdp, policy, filter, init_dist)

for (s, b, a, r, sp, op) in hist
    println("s: $s, action: $a, obs: $op")
end
println("Total reward: $(discounted_reward(hist))")

for s in states(momdp)
    @printf("State(desired_autonomy=%s, performance=%s, given_autonomy=%s, engagement=%s) = Action(give_autonomy=%s)\n", s.desired_autonomy, s.performance, s.given_autonomy, s.engagement, action(policy, ParticleCollection([s])))
end

println(QMDP.alphas(policy))






