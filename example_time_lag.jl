using LinearAlgebra
using StaticArrays
using Distributions
using PyPlot
using TimerOutputs
const Vec2f = SVector{2, Float64}
const Vec4f = SVector{4, Float64}
const VecFrame = SVector{18, Float64}

include("src/geometry/geometry.jl")
include("src/world.jl")
include("src/double_integrator.jl")
include("src/fmt_cvae.jl")
include("read_scenario.jl")
include("predict.jl")
include("PyCallModel.jl")
include("PyCallPredict.jl")

lane1_c = 7.231027419583393
lane2_c = 18.84918099161813
lane3_c = 29.990037979406054
lane4_c = 40.99582664476257
lane5_c = 53.099915981198585
lane_c = [7.231027419583393, 18.84918099161813, 29.990037979406054,
          40.99582664476257, 53.099915981198585]
lane12 = (lane1_c + lane2_c)/2
lane23 = (lane2_c + lane3_c)/2
lane34 = (lane3_c + lane4_c)/2
lane45 = (lane4_c + lane5_c)/2
lane5edge = lane5_c + (lane5_c - lane45)

#us101_1_cluster.7 no case 35
#us101_1_cluster.9 no case 28
#us101_3_cluster.9 no case 3

#us101_2_cluster.9 case 45 invalid sample = 500

#us101_2_cluster.9 case 7 not easy to reach sample = 100
#us101_2_cluster.9 case 28, 38 not easy to reach sample = 100
#us101_2_cluster.9 case 45 not reach sample = 100
#us101_2_cluster.10 case 30 not easy to reach sample = 100

#us101_3_cluster.12 case 11 not easy to reach sample = 100

case_id = 10
case_name = "./us101_3/cluster_11/test_case_file"*string(case_id)*".csv"
host_file_name = "./us101_3/cluster_11/test_host_file"*string(case_id)*".csv"
global scenario, start_frame = read_scenario(case_name)
global lane1_c, lane2_c, lane3_c, lane4_c, lane5_c, lane12, lane23, lane34, lane45, lane5edge

global init_goal = get_init_goal(host_file_name)

global s_init = init_goal[1]
global s_goal = init_goal[4]
global n_goal = 2

frame_id = 3
computing_time = 0.023
curr_t = 0
global lane_id = 0
global left_prob = 0
global keep_prob = 0
global right_prob = 0

result_f = open("./time_lag_exp/test.txt", "a+")
this_traj = open("./time_lag_exp/waypoints"*string(frame_id)*".txt", "a+")
last_traj_file = "./time_lag_exp/waypoints"*string(frame_id-3)*".txt"
reset_timer!()

println("frame id:", frame_id)
global to
global start_frame
global s_goal
global isOK = true
global new_start = false
if frame_id == 0
    s_init = init_goal[1]
    s_goal = init_goal[4]
    if 0 < s_init[1] < lane12
        global lane_id = 1
    elseif lane12 < s_init[1] < lane23
        global lane_id = 2
    elseif lane23 < s_init[1] < lane34
        global lane_id = 3
    elseif lane34 < s_init[1] < lane45
        global lane_id = 4
    elseif lane45 < s_init[1] < lane5edge
        global lane_id = 5
    end
    # println(length(frame))
    global prev_frame = get_frame(scenario, frame_id-1, start_frame)
    global frame1 = get_frame(scenario, frame_id, start_frame)
    #global frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10 = predict_action(prev_frame, frame1, frame_id+start_frame)
    global frame2 = get_frame(scenario, frame_id+1, start_frame)
    global frame3 = get_frame(scenario, frame_id+2, start_frame)
    global frame4 = get_frame(scenario, frame_id+3, start_frame)
    global frame5 = get_frame(scenario, frame_id+4, start_frame)
    global frame6 = get_frame(scenario, frame_id+5, start_frame)
    global frame7 = get_frame(scenario, frame_id+6, start_frame)
    global frame8 = get_frame(scenario, frame_id+7, start_frame)
    global frame9 = get_frame(scenario, frame_id+8, start_frame)
    global frame10 = get_frame(scenario, frame_id+9, start_frame)

    center = Vec2f(s_init[1], s_init[2])
    # println(length(frame))
    global neighbors1, front_neighbors1 = find_neighbors(frame1, center, lane_id, 150.0)
    global neighbors2, front_neighbors2 = find_neighbors(frame2, center, lane_id, 150.0)
    global neighbors3, front_neighbors3 = find_neighbors(frame3, center, lane_id, 150.0)
    global neighbors4, front_neighbors4 = find_neighbors(frame4, center, lane_id, 150.0)
    global neighbors5, front_neighbors5 = find_neighbors(frame5, center, lane_id, 150.0)
    global neighbors6, front_neighbors6 = find_neighbors(frame6, center, lane_id, 150.0)
    global neighbors7, front_neighbors7 = find_neighbors(frame7, center, lane_id, 150.0)
    global neighbors8, front_neighbors8 = find_neighbors(frame8, center, lane_id, 150.0)
    global neighbors9, front_neighbors9 = find_neighbors(frame9, center, lane_id, 150.0)
    global neighbors10, front_neighbors10 = find_neighbors(frame10, center, lane_id, 150.0)
    # println(length(neighbors))
    obstacle_set = set_obstacles(neighbors1)
else
    cum_time = Float64[]
    cum_time_f = open("./time_lag_exp/cum_time0.txt","r")
    n = countlines(cum_time_f)
    seekstart( cum_time_f )
    println(n)
    for i = 1:n
        line = readline(cum_time_f)
        if line != ""
            push!(cum_time, parse(Float64, line))
        end
    end
    waypoints = Vec4f[]
    waypoints_f = open(last_traj_file,"r")
    n = countlines(waypoints_f)
    seekstart( waypoints_f )
    for i = 1:n
        line = split( readline( waypoints_f ), "," )
        if line[1] == ""
            continue
        end
        p = Vec4f(parse(Float64, line[1]),
                parse(Float64, line[2]),
                parse(Float64, line[3]),
                parse(Float64, line[4]))
        push!(waypoints, p)
    end
    global traj, future_wy, reach = get_next_position(cum_time, waypoints, 0.3+computing_time, s_goal)
    if !reach
        trajvalid1 = check_traj(cum_time, waypoints, 0.1, frame_id+1, s_goal, scenario, start_frame)
        trajvalid2 = check_traj(cum_time, waypoints, 0.2, frame_id+2, s_goal, scenario, start_frame)
        trajvalid3 = check_traj(cum_time, waypoints, 0.3, frame_id+3, s_goal, scenario, start_frame)

        if !trajvalid1
            println("traj1 invalid")
        end
        if !trajvalid2
            println("traj2 invalid")
        end
        if !trajvalid3
            println("traj3 invalid")
        end
    end

    if reach
        println("congratulations!!!successful！！！")
        for p in traj
            write(result_f, string(p[1]))
            write(result_f, ',')
            write(result_f, string(p[2]))
            write(result_f, ',')
            write(result_f, string(p[3]))
            write(result_f, ',')
            write(result_f, string(p[4]))
            write(result_f, '\n')
        end
        print_timer()
    end
    global s_init = traj[end]
    print("init:", s_init)
    print("goal:", s_goal)
    println("goal distance:", sqrt((s_init[1]-s_goal[1])^2+(s_init[2]-s_goal[2])^2))
    if sqrt((s_init[1]-s_goal[1])^2+(s_init[2]-s_goal[2])^2+(s_init[3]-s_goal[3])^2+(s_init[4]-s_goal[4])^2) < 10
        println("congratulations!!!successful！！！")
        exit()
    end
    global prev_frame = get_frame(scenario, frame_id-1, start_frame)
    global frame1 = get_frame(scenario, frame_id, start_frame)
    # global frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10 = predict_action(prev_frame, frame1, frame_id+start_frame)
    global frame2 = get_frame(scenario, frame_id+1, start_frame)
    global frame3 = get_frame(scenario, frame_id+2, start_frame)
    global frame4 = get_frame(scenario, frame_id+3, start_frame)
    global frame5 = get_frame(scenario, frame_id+4, start_frame)
    global frame6 = get_frame(scenario, frame_id+5, start_frame)
    global frame7 = get_frame(scenario, frame_id+6, start_frame)
    global frame8 = get_frame(scenario, frame_id+7, start_frame)
    global frame9 = get_frame(scenario, frame_id+8, start_frame)
    global frame10 = get_frame(scenario, frame_id+9, start_frame)

    global center = Vec2f(s_init[1], s_init[2])
    if 0 < s_init[1] < lane12
        global lane_id = 1
    elseif lane12 < s_init[1] < lane23
        global lane_id = 2
    elseif lane23 < s_init[1] < lane34
        global lane_id = 3
    elseif lane34 < s_init[1] < lane45
        global lane_id = 4
    elseif lane45 < s_init[1] < lane5edge
        global lane_id = 5
    end
    global neighbors1, front_neighbors1 = find_neighbors(frame1, center, lane_id, 150.0)
    global neighbors2, front_neighbors2 = find_neighbors(frame2, center, lane_id, 150.0)
    global neighbors3, front_neighbors3 = find_neighbors(frame3, center, lane_id, 150.0)
    global neighbors4, front_neighbors4 = find_neighbors(frame4, center, lane_id, 150.0)
    global neighbors5, front_neighbors5 = find_neighbors(frame5, center, lane_id, 150.0)
    global neighbors6, front_neighbors6 = find_neighbors(frame6, center, lane_id, 150.0)
    global neighbors7, front_neighbors7 = find_neighbors(frame7, center, lane_id, 150.0)
    global neighbors8, front_neighbors8 = find_neighbors(frame8, center, lane_id, 150.0)
    global neighbors9, front_neighbors9 = find_neighbors(frame9, center, lane_id, 150.0)
    global neighbors10, front_neighbors10 = find_neighbors(frame10, center, lane_id, 150.0)
    global obstacle_set = set_obstacles(neighbors1)
    global x_min, x_max, v_min, v_max = get_boundary(neighbors1, center, lane_id, s_goal)
    global W = World(x_min, x_max, v_min, v_max, obstacle_set)

    for p in traj
        write(result_f, string(p[1]))
        write(result_f, ',')
        write(result_f, string(p[2]))
        write(result_f, ',')
        write(result_f, string(p[3]))
        write(result_f, ',')
        write(result_f, string(p[4]))
        write(result_f, '\n')
    end
end

x_min, x_max, v_min, v_max = get_boundary(neighbors1, center, lane_id, s_goal)
W = World(x_min, x_max, v_min, v_max, obstacle_set)
# show(W)
# scatter(s_init[1], s_init[2])
# scatter(s_goal[1], s_goal[2])
# savefig("./us101_3_cluster_100/cluster10/test_0_world.png")
# exit()

Nsample = 110
right_prob = 1/3
left_prob = 1/3
keep_prob = 1/3
println("init:", s_init)
println("goal:", s_goal)
frames = [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10]
fmt = @time FMTree(s_init, s_goal, Nsample, W, right_prob, left_prob, keep_prob, lane_id, frames)
println(length(fmt.Pset))
with_savefig = false
# println("fmt.s_init:", fmt.s_init)
@timeit "solve "*string(frame_id) idx_solution = solve(fmt, frame_id, with_savefig)
clf()
show(fmt)

t = Float64[]
waypoints = Vec4f[]
for idx in idx_solution
    global waypoints
    s0 = fmt.Pset[fmt.parent[idx]]
    s1 = fmt.Pset[idx]
    tau = fmt.time[idx]
    println("s0:", s0)
    println("s1:", s1)
    # println(tau)
    points = show_trajectory(s0, s1, tau, 20, :blue, 1.5)
    if idx != 1
        waypoints = vcat(points, waypoints)
    end
    # print(waypoints)
    scatter(s0[1], s0[2], c=:black, s=10)
    # scatter(s1[1], s1[2], c=:red, s=10)
end
global waypoints
global cum_time = cal_time(waypoints)

for p in waypoints
    write(this_traj, string(p[1]))
    write(this_traj, ',')
    write(this_traj, string(p[2]))
    write(this_traj, ',')
    write(this_traj, string(p[3]))
    write(this_traj, ',')
    write(this_traj, string(p[4]))
    write(this_traj, '\n')
end

time_file = open("./time_lag_exp/cum_time"*string(frame_id)*".txt","a+")
for t in cum_time
    write(time_file, string(t))
    write(time_file, '\n')
end

savefig("./time_lag_exp/test_case_"*string(frame_id)*".png")

open("./time_lag_exp/mytime"*string(frame_id)*".txt","a+") do io
    print_timer(io, TimerOutputs.get_defaulttimer(), allocations=false)
end
exit()
