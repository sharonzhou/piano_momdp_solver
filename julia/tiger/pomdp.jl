struct TigerPOMDP <: POMDP{Bool, Symbol, Bool} # POMDP{State, Action, Observation} all parametarized by Int64s
    r_listen::Float64 # reward for listening (default -1)
    r_findtiger::Float64 # reward for finding the tiger (default -100)
    r_escapetiger::Float64 # reward for escaping (default 10)
    p_listen_correctly::Float64 # prob of correctly listening (default 0.85)
    discount_factor::Float64 # discount
end
# default constructor
function TigerPOMDP()
    return TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)
end;

pomdp = TigerPOMDP()

example_state = false # tiger is hiding behind right door
example_action = :listen # agent listens, can be :openl or :openr
example_observation = true # agent heard the tiger behind the left door

POMDPs.states(::TigerPOMDP) = [true, false];

POMDPs.actions(::TigerPOMDP) = [:openl, :openr, :listen] # default
POMDPs.actions(pomdp::TigerPOMDP, state::Bool) = POMDPs.actions(pomdp) # convenience (actions do not change in different states)
function POMDPs.action_index(::TigerPOMDP, a::Symbol)
    if a==:openl
        return 1
    elseif a==:openr
        return 2
    elseif a==:listen
        return 3
    end
    error("invalid TigerPOMDP action: $a")
end;

# function returning observation space
POMDPs.observations(::TigerPOMDP) = [true, false];
POMDPs.observations(pomdp::TigerPOMDP, s::Bool) = observations(pomdp);

# distribution type that will be used for both transitions and observations
type TigerDistribution
    p::Float64
    it::Vector{Bool}
end
TigerDistribution() = TigerDistribution(0.5, [true, false])
POMDPs.iterator(d::TigerDistribution) = d.it

# transition and observation pdf
function POMDPs.pdf(d::TigerDistribution, so::Bool)
    so ? (return d.p) : (return 1.0-d.p)
end;

# samples from transition or observation distribution
POMDPs.rand(rng::AbstractRNG, d::TigerDistribution) = rand(rng) <= d.p;

# Resets the problem after opening door; does nothing after listening
function POMDPs.transition(pomdp::TigerPOMDP, s::Bool, a::Symbol)
    d = TigerDistribution()
    if a == :openl || a == :openr
        d.p = 0.5
    elseif s
        d.p = 1.0
    else
        d.p = 0.0
    end
    d
end;

# reward model
function POMDPs.reward(pomdp::TigerPOMDP, s::Bool, a::Symbol)
    r = 0.0
    a == :listen ? (r+=pomdp.r_listen) : (nothing)
    if a == :openl
        s ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    end
    if a == :openr
        s ? (r += pomdp.r_escapetiger) : (r += pomdp.r_findtiger)
    end
    return r
end
POMDPs.reward(pomdp::TigerPOMDP, s::Bool, a::Symbol, sp::Bool) = reward(pomdp, s, a);

# observation model
function POMDPs.observation(pomdp::TigerPOMDP, a::Symbol, s::Bool)
    d = TigerDistribution()
    pc = pomdp.p_listen_correctly
    if a == :listen
        s ? (d.p = pc) : (d.p = 1.0-pc)
    else
        d.p = 0.5
    end
    d
end;

POMDPs.discount(pomdp::TigerPOMDP) = pomdp.discount_factor
POMDPs.n_states(::TigerPOMDP) = 2
POMDPs.n_actions(::TigerPOMDP) = 3
POMDPs.n_observations(::TigerPOMDP) = 2;

POMDPs.initial_state_distribution(pomdp::TigerPOMDP) = TigerDistribution(0.5, [true, false]);






