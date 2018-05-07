using BasicPOMCP
using POMDPToolbox

pomdp = BabyPOMDP()
solver = POMCPSolver()
planner = solve(solver, pomdp)

hist = simulate(HistoryRecorder(max_steps=10), pomdp, planner);
println("reward: $(discounted_reward(hist))")