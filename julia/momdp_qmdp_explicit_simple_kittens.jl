# Transition function P(s' | a, o, b)
# Slide 16: https://www.cs.cmu.edu/~ggordon/780-fall07/lectures/POMDP_lecture.pdf
function transition(m::MOMDP, s::State, a::Symbol, o::Obs, b::Any)
    # P(o | s', a, b) = P(o | s') = p(s' | o) * p(o) / p(s')
    # for each next state, given the performance + duration, what's its probability given the observation? 
        # 0 for most states except the relevant latent ones
        # we have P(u' | u, p, gu) -- is *u* a belief? b/c not certain -- and P(i' | u', gu)
        # we have gu, p, i', essentially want P(gu, p, i' | s') = P(i' | u')
    if a == :give_autonomy
        given_autonomy = true
        if o.duration
            p_o_given_sp_desired = m.p_engaged_when_desired_given
            p_o_given_sp_not_desired = m.p_engaged_when_not_desired_given
        else
            p_o_given_sp_desired = 1.0 - m.p_engaged_when_desired_given
            p_o_given_sp_not_desired = 1.0 - m.p_engaged_when_not_desired_given
        end
    else
        given_autonomy = false
        if o.duration
            p_o_given_sp_desired = m.p_engaged_when_desired_not_given
            p_o_given_sp_not_desired = m.p_engaged_when_not_desired_not_given
        else
            p_o_given_sp_desired = 1.0 - m.p_engaged_when_desired_not_given
            p_o_given_sp_not_desired = 1.0 - m.p_engaged_when_not_desired_not_given
        end
    end

    # P(s' | a, b) = \sum_s p(s' | a, s) * b(s)
    p_sp_given_ab_desired = sum(unroll_sparse_cat(transition(m, State(true, s.performance, s.given_autonomy, s.engagement), a)) .* b)
    p_sp_given_ab_not_desired = sum(unroll_sparse_cat(transition(m, State(false, s.performance, s.given_autonomy, s.engagement), a)) .* b)

    # P(o | a, b) = \sum_s' p(o | s') * p(s' | a, b)
    p_o_given_ab_desired = sum(p_sp_given_ab_desired * p_sp_given_ab_desired)
    p_o_given_ab_not_desired = sum(p_sp_given_ab_not_desired * p_sp_given_ab_not_desired)

    sp_desired_autonomy = true
    sps = State[]
    probs = Float64[]
    push!(sps, State(sp_desired_autonomy, o.performance, given_autonomy, o.duration))
    push!(probs, p_o_given_sp_desired * p_sp_given_ab_desired \ p_o_given_ab_desired)
    push!(sps, State(!sp_desired_autonomy, o.performance, given_autonomy, o.duration))
    push!(probs, p_o_given_sp_not_desired * p_sp_given_ab_not_desired \ p_o_given_ab_not_desired)
    
    # Normalize probs
    println("Probs before normalization: ", probs)
    probs = probs / sum(probs)
    println("Probs after normalization: ", probs)

    # Debugging
    print("\n######\n")
    print(s, " desired_autonomy, performance, given_autonomy, engagement\n", a, "\n")
    print(sps, "\n")
    print(probs, "\n")
    print("\n######\n")
    return SparseCat(sps, probs)
end




# Visualize D3tree setup
text = ["revoke"]
tree = [[2,3,4,5]]
node_idx = 6
for node in tree
    # depth d requires # nodes = 2*(4^0+4^1+4^2+4^3+...4^d) (mult by 2 b/c it's num expansion nodes + num extra action nodes)
    # d=30 is 3.0744573e+18 (copy paste below for chrome calculator evaluation)
    # 2*(4^0+4^1+4^2+4^4+4^5+4^6+4^7+4^8+4^9+4^10+4^11+4^12+4^13+4^14+4^15+4^16+4^17+4^18+4^19+4^20+4^21+4^22+4^23+4^24+4^25+4^26+4^27+4^28+4^29+4^30)=
    if node_idx < 3.0744573e+18 
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

for seed = 1:1000
    hist = HistoryRecorder(max_steps=30, rng=MersenneTwister(seed), show_progress=true)
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



hist_tree = Dict()
for seed = 1:5
    println(seed)
    hist = HistoryRecorder(max_steps=4, rng=MersenneTwister(seed), show_progress=true)
    hist = simulate(hist, momdp, policy, filter, init_dist)

    hist_ao = String[]
    hist_b = Array[]
    i = 1
    for (s, b, a, r, o, sp, bp) in eachstep(hist, "(s, b, a, r, o, sp, bp)")
        if a == :give_autonomy
            push!(hist_ao, "give")
            action = "give"
        else
            push!(hist_ao, "revoke")
            action = "revoke"
        end

        if o.performance
            if o.duration
                push!(hist_ao, "good eng")
                obs = "good eng"
            else
                push!(hist_ao, "good dis")
                obs = "good dis"
            end
        else
             if o.duration
                push!(hist_ao, "bad eng")
                obs = "bad eng"
            else
                push!(hist_ao, "bad dis")
                obs = "bad dis"
            end
        end

        belief = QMDP.belief_vector(policy, b)
        
        if haskey(hist_tree, i)
            if haskey(hist_tree[i], action)
                if haskey(hist_tree[i][action], obs)
                    # pass; already recorded
                else
                    hist_tree[i][action][obs] = belief
                end
            else
                hist_tree[i][action] = Dict()
                hist_tree[i][action][obs] = belief
            end
        else
            hist_tree[i] = Dict()
            hist_tree[i][action] = Dict()
            hist_tree[i][action][obs] = belief
        end
        
        push!(hist_b, QMDP.belief_vector(policy, b))

        i += 1
    end
    # println("here")
    # println(hist_ao)
    # println(hist_b)
    # If not previously explored
    # if hist_ao in hists_aos
        # continue
    # else
    push!(hists_aos, hist_ao)

    hist_dict = Dict("ao" => hist_ao, "belief" => hist_b)
    push!(hists, hist_dict)
    # end
end
println(hists_aos)
println(hists)



# Individual simulation
seed = 5
hist = HistoryRecorder(max_steps=4, rng=MersenneTwister(seed), show_progress=true)
hist = simulate(hist, momdp, policy, filter, init_dist)
for (s, b, a, r, o, sp, bp) in eachstep(hist, "(s, b, a, r, o, sp, bp)")
    print("####\n")
    println(QMDP.belief_vector(policy, b))
    println("s(desire, perf, given, eng): $s")
    println("a: $a") 
    println("r: $r")
    println("o(perf, given, dur): $o") 
    println("sp: $sp")
    println(QMDP.belief_vector(policy, bp)) 
end
for s in states(momdp)
    # @show s
    @show action(policy, ParticleCollection([s]))
    @printf("State(desired_autonomy=%s, performance=%s, given_autonomy=%s, engagement=%s) = Action(give_autonomy=%s)\n", s.desired_autonomy, s.performance, s.given_autonomy, s.engagement, action(policy, ParticleCollection([s])))
end

println(QMDP.alphas(policy))



hist = sim(momdp, max_steps=4) do obs
    println("Observation was $obs.")
    return 1
end

children = []
for (s, b, a, r, sp, op) in hist
    print("####\n")
    println("s(desired_autonomy, performance, given_autonomy, engagement): $s, action: $a, obs(performance, given_autonomy, duration): $op")
    # push!(children, op)

    # println(QMDP.value(policy, b))
    # println(QMDP.action(policy, b))
    println(QMDP.belief_vector(policy, b))
    # println(QMDP.unnormalized_util(policy, b))
    # println(policy.action_map)
    # println("s: $s, b: $(b.b), action: $a, obs: $op")
end
println("Total reward: $(discounted_reward(hist))")
print(states(momdp))

print(children)
print(typeof(children))
tree = D3Trees(children)
print(tree)


for (action_true, action_false) in policy.alphas
    println("$action_true")
end
using D3Trees

D3Tree(policy, State(true, true, true, true), init_expand=2)

POMCPSolver

solver = POMCPSolver()
policy = solve(solver, momdp)
# print(policy)

for (s, a, o) in stepthrough(momdp, policy, "sao", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end


Generative model (cannot use for QMDP)
function generate_s(m::MOMDP, s::State, a::Act, rng::AbstractRNG)
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







