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
    # given_difficulty::Bool
end

struct Act 
    give_autonomy::Bool
    give_difficulty::Bool
end

struct Obs
    duration::Bool #TODO: Float64 or Int64

    performance::Bool #TODO: Float64 or Int64
    # exercised_autonomoy::Bool
    # exercised_difficulty::Bool
    given_autonomy::Bool
    # given_difficulty::Bool 
end

# Connect observation space to state space for observable variables
Obs(s::State, d::Bool) = Obs(d, s.performance, s.given_autonomoy)

struct MOMDP <: POMDP{State, Act, Obs}
    p_just_right_when_desired_good_given::Float64
    p_just_right_when_desired_good_not_given::Float64
    p_just_right_when_desired_bad_given::Float64
    p_just_right_when_desired_bad_not_given::Float64
    p_just_right_when_not_desired_good_given::Float64
    p_just_right_when_not_desired_good_not_given::Float64
    p_just_right_when_not_desired_bad_given::Float64
    p_just_right_when_not_desired_bad_not_given::Float64
    # discount::Float64
end

# From CPT
MOMDP() = MOMDP(0.99, 0.9, 0.3, 0.8, 0.8, 0.1, 0.01, 0.2)

# discount(m::MOMDP) = m.discount

function generate_s(m::MOMDP, s::State, a::Obs, rng::AbstractRNG)
    if s.desired_autonomy
        if s.performance
            if s.given_autonomy
                return rand(rng) < m.p_just_right_when_desired_performed_given ? true : false
            else
                return rand(rng) < m.p_just_right_when_desired_performed_not_given ? true : false
            end
        else
            if s.given_autonomy
                return rand(rng) < m.p_just_right_when_desired_bad_given ? true : false
            else
                return rand(rng) < m.p_just_right_when_desired_bad_given ? true : false
            end
        end
    else
        if s.performance
            if s.given_autonomy
                return rand(rng) < m.p_just_right_when_not_desired_performed_given ? true : false
            else
                return rand(rng) < m.p_just_right_when_not_desired_performed_not_given ? true : false
            end
        else
            if s.given_autonomy
                return rand(rng) < m.p_just_right_when_not_desired_bad_given ? true : false
            else
                return rand(rng) < m.p_just_right_when_not_desired_bad_given ? true : false
            end
        end
    end
end

function generate_o(m::MOMDP, s::State, a::Obs, sp::State, rng::AbstractRNG)
    if sp # hungry
        return rand(rng) < m.p_cry_when_hungry ? true : false
    else # not hungry
        return rand(rng) < m.p_cry_when_not_hungry ? true : false
    end
end

# r_hungry
reward(m::MOMDP, s::State, a::Act) = (s ? p.r_hungry : 0.0) + (a ? p.r_feed : 0.0)

initial_state_distribution(p::BabyPOMDP) = [false] # note rand(rng, [false]) = false, so this is encoding that the baby always starts out full