importall POMDPs

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
end

struct Act 
    give_autonomy::Bool
    give_difficulty::Bool
end

struct Obs
    duration::Bool #TODO: Float64 or Int64

    performance::Bool #TODO: Float64 or Int64
    exercised_autonomoy::Bool
    exercised_difficulty::Bool 
end

# Connect observation space to state space for observable variables
Obs(s::State, d::Bool) = Obs(d, s.performance, s.exercised_autonomoy, s.exercised_difficulty)

struct MOMDP <: POMDP{State, Act, Obs}
    r_continue::Int64
    r_complete::Int64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    # discount::Float64
end
BabyPOMDP() = BabyPOMDP(-5., -10., 0.1, 0.8, 0.1, 0.9)

discount(p::BabyPOMDP) = p.discount

function generate_s(p::BabyPOMDP, s::Bool, a::Bool, rng::AbstractRNG)
    if s # hungry
        return true
    else # not hungry
        return rand(rng) < p.p_become_hungry ? true : false
    end
end

function generate_o(p::BabyPOMDP, s::Bool, a::Bool, sp::Bool, rng::AbstractRNG)
    if sp # hungry
        return rand(rng) < p.p_cry_when_hungry ? true : false
    else # not hungry
        return rand(rng) < p.p_cry_when_not_hungry ? true : false
    end
end

# r_hungry
reward(m::MOMDP, s::State, a::Act) = (s ? p.r_hungry : 0.0) + (a ? p.r_feed : 0.0)

initial_state_distribution(p::BabyPOMDP) = [false] # note rand(rng, [false]) = false, so this is encoding that the baby always starts out full