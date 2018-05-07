importall POMDPs

# state: true=hungry, action: true=feed, obs: true=crying

struct BabyPOMDP <: POMDP{Bool, Bool, Bool}
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    discount::Float64
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
reward(p::BabyPOMDP, s::Bool, a::Bool) = (s ? p.r_hungry : 0.0) + (a ? p.r_feed : 0.0)

initial_state_distribution(p::BabyPOMDP) = [false] # note rand(rng, [false]) = false, so this is encoding that the baby always starts out full