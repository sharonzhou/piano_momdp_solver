# the scenario 

#  |-------------–†---|     † --- pedestrian trying to cross
#  |        |         |     x --- occluding vehicle on the road
#  |        |         |     R --- robot vehicle that is occluded
#  |        |   X     |
#  |        |         |     Robot vehicle get's observation of pedestrian's location unless pedestrian is occluded.
#  |        |         |     In this scenario, no observation should be returned. The POMDP tries to find a policy
#  |     R  |         |     /controller to get the robot vehicle safely accross crosswalk without hitting pedestrian.
#  |        |         |     For now, we aassume sensors are always accurate unless occluded in which case they give no
#  |        |         |     reading. 
#  |        |         |


importall POMDPs

# first import the POMDPs.jl interface
using POMDPs

# import our helper Distributions.jl module
using Distributions 

# POMDPToolbox has some glue code to help us use Distributions.jl
using POMDPToolbox

# set of possible pedestrian trajectories
trajectories = [8 8 8 8 8 8 8 8 8 8; # don't cross
            8 7 7 7 7 7 7 7 7 7; # show up then stop
            8 7 7 7 6 5 4 2 1 0; # move stop and go
            8 8 7 6 5 4 3 2 1 0] # boldly cross

struct DrivingState
    
    x_r::Float64 
    y_r::Float64 
    dy_r::Float64 
    x_o::Float64
    y_o::Float64 
    dy_o::Float64
    x_p::Float64 
    y_p::Float64 

    sim_time::Int64   # simulation time, useful for debugging, plotting and rolling out pedestrian trajectory
    c::Int64          # choice of pedestrian from 4 possible trajectories
        
end



# default constructor for driving state
DrivingState() = DrivingState(2.0,0.0,1.0,5.0,0.0,1.0,8.0,8.0, 0,rand([1 2 3 4]))

mutable struct DrivingPOMDP <: POMDP{DrivingState, Int64, Nullable{DrivingState}}
    
    p_cross::Float64
    discount::Float64
    
    n_y_r::Int64 
    n_dy_r::Int64 
    n_y_o::Int64 
    n_dy_o::Int64
    n_x_p::Int64
    
   
       
end




# helper function to check if two driving states are equal 
function isEqual(lhs::DrivingState, rhs::DrivingState)
    
    res::Bool = false
    res = (lhs.x_r == rhs.x_r && lhs.y_r == rhs.y_r && lhs.dy_r == rhs.dy_r && lhs.x_o == rhs.x_o && lhs.y_o == rhs.y_o && lhs.dy_o == rhs.dy_o && lhs.x_p == rhs.x_p && lhs.y_p == rhs.y_p
&& lhs.sim_time == rhs.sim_time && lhs.c == rhs.c)
    
    return res
    
    
end



# default constructors
DrivingPOMDP() = DrivingPOMDP(1.0, 0.9999999, 11,6,11,6,9, rand([1 2 3 4]) ,0,0)



function POMDPs.states(mdp::DrivingPOMDP)
    s = DrivingState[] # initialize an array of possible states, hardcoded boundaries for mow
    # loop over all our states:
    for x_r = 2, y_r = 0:10, dy_r = 0:5, x_o = 5, y_o = 0:10, dy_o = 0:5, x_p = 0:8, y_p = 8, sim_time = 0:9, c=1:4
        push!(s, DrivingState(x_r,y_r,dy_r, x_o, y_o, dy_o, x_p, y_p, sim_time, c))
    end
    return s
end;


# given position of obstacle vehicle, position of pedestrian and position of robot vehicle, determines if 
# the pedestrian is occluded from the view of the robot vehicle by the obstacle vehicle.
function is_occluded(x_r::Float64, y_r::Float64, x_o::Float64, y_o::Float64, 
    x_p::Float64, y_p::Float64)
    
    flag::Bool = false
    
    # get angle of fov assuming the obstacle is of length 4m 
    theta_1 = atan2(((y_o-2)-y_r), (x_o-x_r))
    theta_2 = atan2(((y_o+2)-y_r), (x_o-x_r))
    theta_p = atan2(((y_p-y_r)),(x_p-x_r))
    
    if ((theta_p >= theta_1) && (theta_p <= theta_2) && (x_p >= x_o))
        
        flag = true
        
    end
    
    return flag
    
    
end


# helper function for generation plots
function get_slope_intercept(x_r::Float64, y_r::Float64, x_o::Float64, y_o::Float64)
    
    b = [y_r;y_o]
    A = [x_r 1;x_o 1]
    
    res = inv(A)*b
    
    m = res[1]
    c = res[2]
    
    return m,c
    
end

# fucntion to generate pedestrian's next position
function transition_pedestrian(x_p::Float64, y_p::Float64, c::Int64, sim_time::Int64,rng::AbstractRNG)

    end_of_cw::Float64 = 0.0
    delta_x::Float64 = 1.0
            
    x_p_new::Float64 = 0.0
    y_p_new::Float64 = 0.0
            
    p_cross::Float64 = 0.5
    
    
    tmp = mod(sim_time +1, 10)
#     trajectories = [8 8 8 8 8 8 8 8 8 8; # don't cross
#                     8 7 7 7 7 7 7 7 7 7; # show up then stop
#                     8 7 7 7 6 5 4 2 1 0; # move stop and go
#                     8 8 7 6 5 4 3 2 1 0] # boldly cross
    
    
    if ((tmp == 0 )  && (sim_time == 9) ) # reset if previous time was 9 and were are at 0 now
        
        x_p_new = 8.0
        y_p_new = y_p
        c_new = rand([1 2 3 4]) 
        
    
    else
            

            x_p_new = trajectories[c, tmp+1] # array indexing starts from 1
            y_p_new = y_p
            c_new = c

                        
    end
    
    if(x_p_new <= 0) # prevent from moving out of view
        
        x_p_new = 0
        
    end
    
    return x_p_new, y_p_new, c_new, mod(tmp, 10)
    


end


# function for generating obstacle vehicle's next position
function transition_obstacle(x_o::Float64, y_o::Float64,x_p::Float64, y_p::Float64,rng::AbstractRNG)
    
    delta_x::Float64 = 0.0
    delta_y::Float64 = 2.0
    
    ped_tolerance::Float64 = delta_y
    
    x_o_new::Float64 = 0.0
    y_o_new::Float64 = 0.0
    dy_o_new::Float64 = 0.0
    
    if (y_o >= 10.0) # reset
        
        y_o_new = 0.0
        x_o_new = x_o
        dy_o_new = delta_y
        
    else
        
        if ((abs(y_p - y_o) <= ped_tolerance) && (abs(x_p - x_o) <= 2.0)) # stop too close to pedestrian
            
            x_o_new = x_o
            y_o_new = y_o
            dy_o_new = 0.0
            
        else
            
            x_o_new = x_o + delta_x
            y_o_new = y_o + delta_y
            dy_o_new = delta_y
            
        end
        
        
    end
    
    if (y_o_new >= 10.0) # prevent from moving out of view
        
        y_o_new = 10.0
        
    end
    
    if(dy_o_new > 5)
        dy_o_new = 5
    elseif (dy_o_new < 0)
        dy_o_new = 0
    end
    
    return x_o_new, y_o_new, dy_o_new
   
    
    
end

# function for generating robot vehicle's next position
function transition_robot(x_r::Float64, y_r::Float64,dy_r::Float64,action::Int64,rng::AbstractRNG)
    
    delta_y_prime::Float64 = 1.0
    
    y_r_new::Float64 = 0.0
    x_r_new::Float64 = 0.0
    y_r_prime::Float64 = 1.0
    
    if (y_r >= 10) # reset
        
        y_r_new = 0.0
        x_r_new = x_r
        y_r_prime = 1.0
        
    else
        
        if(action == 1) # speeding up
            
            y_r_prime = dy_r + delta_y_prime
            if(y_r_prime > 5)
                y_r_prime = 5
            end
            x_r_new = x_r 
            y_r_new = y_r + y_r_prime
            
        elseif (action == 2) # slowing down 
            
            y_r_prime = dy_r - delta_y_prime
            x_r_new = x_r
            
            if (y_r_prime < 0)
                
                y_r_prime = 0
                
            end
            
            y_r_new = y_r + y_r_prime
            
        elseif (action == 3) # stopped instantaneously
            
            y_r_prime = 0
            x_r_new = x_r
            y_r_new = y_r + y_r_prime
            
        elseif (action == 4) # constant velocity
            
            x_r_new = x_r
            y_r_prime = dy_r
            if(y_r_prime >= 5)
                y_r_prime = 5
            end
            
            y_r_new = y_r + y_r_prime
            
        end
        
        
    end
    
    if (y_r_new >= 10) # prevent from moving out of view
        
        y_r_new = 10.0
        
    end
    
    return x_r_new, y_r_new, y_r_prime
    
end

# reward function
function get_reward(x_r, y_r, dy_r, x_p, y_p, action, sim_time)
    
    reward::Float64 = 0.0
    
    if((x_r == x_p) && (y_r == y_p)) # crashed into pedestrian 
        
        reward = -6000
        
    elseif( (x_p < 7) && ((y_p - y_r) > 0)  &&  ((y_p - y_r) < 5)) # pedestrian on cw and in front of vehicle
        
        predicted_x_r, predicted_y_r, predicted_y_r_prime = transition_robot(x_r, y_r, dy_r, action, MersenneTwister(0))
        
        if(((y_p - predicted_y_r) == 1) && (predicted_y_r_prime == 0))# if you will stop at 1m away, get reward
            
            reward = 4000
            
        elseif ((y_p - predicted_y_r) < 0) # else if will cross cw, punish 
            
            reward = - 5001
            
        end
        
    elseif( (x_p < 7) && ((y_p - y_r) == 0) ) # vehicle on cw with pedestrian 
        
        reward = - 5001
        
    elseif(y_r >= 10.0)
        
        reward = 5000
        
    elseif(dy_r == 0)  # stopping randomly 
        
        reward = -3000
        
    end
    
    return reward
        
    
   
end


# function that returns last position if occluded or true position otherwise --- currently obsolete
function observer(x_r, y_r, x_o_new, y_o_new, x_p, y_p, x_p_new, y_p_new)
    
    flag = is_occluded(x_r, y_r, x_o_new, y_o_new, x_p_new, y_p_new)
    if(flag == true)
        
        
        return  x_p, y_p, true 
        
    end
        
    return x_p_new, y_p_new, false

                            
end

function generate_sor(p::DrivingPOMDP,s::DrivingState, action::Int64, rng::AbstractRNG)
    
    x_r_new::Float64 = 0.0
    y_r_new::Float64 = 0.0
    dy_r_new::Float64 = 0.0
    
    x_o_new::Float64 = 0.0
    y_o_new::Float64 = 0.0
    dy_o_new::Float64 = 0.0
    
    x_p_new::Float64 = 0.0
    y_p_new::Float64 = 0.0
    x_p_new2::Float64 = 0.0
    y_p_new2::Float64 = 0.0
    c_new::Int64 = 5
    
    x_p_old::Float64 = s.x_p
    y_p_old::Float64 = s.y_p
    

    sim_time::Int64 = s.sim_time
    is_occluded::Bool = false
    
    flag::Bool = false
    
    new_reward = get_reward(s.x_r,s.y_r,s.dy_r,s.x_p,s.y_p, action, s.sim_time)
    x_r_new, y_r_new, dy_r_new = transition_robot(s.x_r, s.y_r, s.dy_r, action, rng)
    x_o_new, y_o_new, dy_o_new = transition_obstacle(s.x_o, s.y_o, s.x_p, s.y_p, rng)
    x_p_new, y_p_new, c_new, sim_time = transition_pedestrian(s.x_p, s.y_p, s.c, s.sim_time,rng)
    
    new_state = DrivingState(x_r_new, y_r_new, dy_r_new, x_o_new, y_o_new, dy_o_new, x_p_new, y_p_new, sim_time, c_new)

    
    x_p_new2, y_p_new2, is_occluded = observer(x_r_new, y_r_new, x_o_new, y_o_new, x_p_old, y_p_old, x_p_new,y_p_new)
    
    
    new_observation = Nullable{DrivingState}()
    if (!is_occluded)
        new_observation = Nullable(DrivingState(x_r_new, y_r_new, dy_r_new, x_o_new, y_o_new, dy_o_new, x_p_new2, y_p_new2, sim_time, c_new))
    end
    
    
    return new_state, new_observation, new_reward
    
end

function generate_s(p::DrivingPOMDP,s::DrivingState, action::Int64, rng::AbstractRNG)
    
    x_r_new::Float64 = 0.0
    y_r_new::Float64 = 0.0
    dy_r_new::Float64 = 0.0
    
    x_o_new::Float64 = 0.0
    y_o_new::Float64 = 0.0
    
    x_p_new::Float64 = 0.0
    y_p_new::Float64 = 0.0
    c_new::Int64 = 5

    sim_time::Int64 = s.sim_time
    is_occluded::Int64 = false
    
    x_r_new, y_r_new, dy_r_new = transition_robot(s.x_r, s.y_r, s.dy_r, action, rng)
    x_o_new, y_o_new, dy_o_new = transition_obstacle(s.x_o, s.y_o, s.x_p, s.y_p, rng)
    x_p_new, y_p_new, c_new, sim_time = transition_pedestrian(s.x_p, s.y_p, s.c, s.sim_time, rng)
    
    new_state = DrivingState(x_r_new, y_r_new, dy_r_new, x_o_new, y_o_new, dy_o_new, x_p_new, y_p_new, sim_time, c_new)

    
    return new_state
    
end

# changing it to return null if state is occluded
function generate_so(p::DrivingPOMDP,s::DrivingState, action::Int64, rng::AbstractRNG)
    
    x_r_new::Float64 = 0.0
    y_r_new::Float64 = 0.0
    dy_r_new::Float64 = 0.0
    
    x_o_new::Float64 = 0.0
    y_o_new::Float64 = 0.0
    
    x_p_new::Float64 = 0.0
    y_p_new::Float64 = 0.0
    x_p_new2::Float64 = 0.0
    y_p_new2::Float64 = 0.0
    c_new::Int64 = 5
    
    
    sim_time::Int64 = s.sim_time
    
    is_occluded::Bool = false
    flag::Bool = false
    
    x_p_old::Float64 = s.x_p
    y_p_old::Float64 = s.y_p
    
    new_reward = get_reward(s.x_r,s.y_r,s.dy_r,s.x_p,s.y_p, action, s.sim_time)
    x_r_new, y_r_new, dy_r_new = transition_robot(s.x_r, s.y_r, s.dy_r, action, rng)
    x_o_new, y_o_new, dy_o_new = transition_obstacle(s.x_o, s.y_o, s.x_p, s.y_p, rng)
    x_p_new, y_p_new, c_new, sim_time = transition_pedestrian(s.x_p, s.y_p, s.c, s.sim_time,  rng)
    
    new_state = DrivingState(x_r_new, y_r_new, dy_r_new, x_o_new, y_o_new, dy_o_new, x_p_new, y_p_new, sim_time, c_new)
    
    x_p_new2, y_p_new2, is_occluded = observer(x_r_new, y_r_new,x_o_new, y_o_new, x_p_old, y_p_old, x_p_new,y_p_new)
    
    new_observation = Nullable{DrivingState}()
    if (!is_occluded)
        new_observation = Nullable(DrivingState(x_r_new, y_r_new, dy_r_new, x_o_new, y_o_new, dy_o_new, x_p_new2, y_p_new2, sim_time, c_new))
    end

    
    return new_state,new_observation
    
end



function POMDPs.state_index(mdp::DrivingPOMDP, state::DrivingState)
    
   
    sub_y_r = state.y_r*mdp.n_dy_r*mdp.n_y_o*mdp.n_dy_o*mdp.n_x_p*10*4
    sub_dy_r = state.dy_r*mdp.n_y_o*mdp.n_dy_o*mdp.n_x_p*10*4
    sub_y_o = state.y_o*mdp.n_dy_o*mdp.n_x_p*10*4
    sub_dy_o = state.dy_o*mdp.n_x_p*10*4
    sub_x_p = state.x_p*10*4
    sub_sim_time = state.sim_time*4
    sub_c = (state.c -1)
    

    
    return convert(Int64,sub_y_r+sub_dy_r+sub_y_o+sub_dy_o+sub_x_p+sub_sim_time+sub_c+1)
end




function POMDPs.observations(mdp::DrivingPOMDP)
    s = DrivingState[] # initialize an array of GridWorldStates
    # loop over all our states:
    for x_r = 2, y_r = 0:10, dy_r = 0:5, x_o = 5, y_o = 0:10, dy_o = 0:5, x_p = 0:8, y_p = 8, sim_time=0:9, c=1:4
        push!(s, DrivingState(x_r,y_r,dy_r, x_o, y_o, dy_o, x_p, y_p, sim_time, c))
    end
    return s
end



POMDPs.observations(mdp::DrivingPOMDP, state::DrivingState) = observations(mdp);

function POMDPs.obs_index(mdp::DrivingPOMDP, state::DrivingState)
    
    sub_y_r = state.y_r*mdp.n_dy_r*mdp.n_y_o*mdp.n_dy_o*mdp.n_x_p*10*4
    sub_dy_r = state.dy_r*mdp.n_y_o*mdp.n_dy_o*mdp.n_x_p*10*4
    sub_y_o = state.y_o*mdp.n_dy_o*mdp.n_x_p*10*4
    sub_dy_o = state.dy_o*mdp.n_x_p*10*4
    sub_x_p = state.x_p*10*4
    sub_sim_time = state.sim_time*4
    sub_c = (state.c -1)
    

    
    return convert(Int64,sub_y_r+sub_dy_r+sub_y_o+sub_dy_o+sub_x_p+sub_sim_time+sub_c+1)
end






actions(::DrivingPOMDP) = [1,2,3,4] # default
actions(pomdp::DrivingPOMDP, state::DrivingState) = POMDPs.actions(pomdp) # convenience (actions do not change in different states)
POMDPs.n_actions(::DrivingPOMDP) = 4
POMDPs.n_states(mdp::DrivingPOMDP) = mdp.n_y_r*mdp.n_dy_r*mdp.n_y_o*mdp.n_dy_o*mdp.n_x_p*10*4
POMDPs.n_observations(mdp::DrivingPOMDP) = mdp.n_y_r*mdp.n_dy_r*mdp.n_y_o*mdp.n_dy_o*mdp.n_x_p*10*4
POMDPs.discount(pomdp::DrivingPOMDP) = pomdp.discount


# pretty much a uniform initial distibution for now.
function get_initial_state_dist()
    full_states = states(DrivingPOMDP())
    initial_dist_states = DrivingState[]
    initial_dist_probs = Float64[]
    for s in full_states
        
        if ((s.x_r == 2.0) && (s.y_r == 0.0) && (s.dy_r == 1.0) && (s.x_o == 5.0) && (s.y_o == 0.0) && (s.dy_o == 1.0) && (s.x_p == 8.0) && (s.y_p == 8.0) && (s.sim_time == 0))

            push!(initial_dist_states, s)
            push!(initial_dist_probs, 1.0)

        end
        n = 1/(sum(initial_dist_probs))
        fill!(initial_dist_probs, n)

    end

    return SparseCat(initial_dist_states, initial_dist_probs)
    
end


initial_state_distribution(pomdp::DrivingPOMDP) = get_initial_state_dist();



function POMDPs.transition(p::DrivingPOMDP,s::DrivingState, action::Int64)
    
    new_state, new_observation = generate_so(p, s, action, MersenneTwister(0)) # check if occlusion stuff works here
    
    adjacent_states = DrivingState[]
    adjacent_probs = Float64[]
    
    if ((s.sim_time == 9) && (new_state.sim_time == 0) ) # time to make random choice about next motion of pedestrian
        
        x_r = new_state.x_r 
        y_r = new_state.y_r
        dy_r = new_state.dy_r 
        x_o = new_state.x_o
        y_o = new_state.y_o 
        dy_o = new_state.dy_o
        x_p = new_state.x_p 
        y_p = new_state.y_p 

        sim_time = new_state.sim_time
        

    
        for c=1:4
            
            tmp_driving_state = DrivingState(x_r, y_r, dy_r, x_o,y_o, dy_o, x_p, y_p, sim_time, c )
            push!(adjacent_states, tmp_driving_state)
            push!(adjacent_probs, 0.25)
            
            
        end
        
    else
    
        push!(adjacent_states, new_observation)
        push!(adjacent_probs, 1.0)
        

        
    end

    
    return SparseCat(adjacent_states, adjacent_probs)
        
end




function POMDPs.observation(mdp::DrivingPOMDP, sp::DrivingState)    
    
    return SparseCat([sp], [1.0])  # observer is very accurate if there's an observation
    
end



function POMDPs.reward(p::DrivingPOMDP,s::DrivingState, action::Int64) #deleted action
    new_state, observation, reward = generate_sor(p,s,action, MersenneTwister(0))
    
    return reward
end

# ---------------------------------------- test code --------------------------------------------------

using POMDPModels
using BasicPOMCP
using POMDPToolbox
using SARSOP
using QMDP

@requirements_info QMDPSolver() DrivingPOMDP()

using BasicPOMCP
using POMDPToolbox
using PyPlot

# load the module
# initialize our driving POMDP


pomdp = DrivingPOMDP()


solver = QMDPSolver(max_iterations=1, tolerance=1e-3)
qmdp_policy = solve(solver, pomdp, verbose=true)


init_dist = initial_state_distribution(pomdp) # initialize distriubtion over state

up = updater(qmdp_policy) # belief updater for our policy
#up = PreviousObservationUpdater()

hist = HistoryRecorder(max_steps=50, rng=MersenneTwister(1)) # history recorder that keeps track of states, observations and beliefs
hist = simulate(hist, pomdp, qmdp_policy, up, init_dist, DrivingState())







