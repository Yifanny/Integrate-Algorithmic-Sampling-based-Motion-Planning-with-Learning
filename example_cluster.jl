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

#sample = 50:
#us101_1_cluster.7 no case 20, 25
for case_id in 1:100
    print("case id: ",case_id, "\n")
    sleep(3)
    case_name = "./us101_2/cluster_10/test_case_file"*string(case_id)*".csv"
    host_file_name = "./us101_2/cluster_10/test_host_file"*string(case_id)*".csv"
    global scenario, start_frame = read_scenario(case_name)
    global lane1_c, lane2_c, lane3_c, lane4_c, lane5_c, lane12, lane23, lane34, lane45, lane5edge

    global init_goal = get_init_goal(host_file_name)

    global s_init = init_goal[1]
    global s_goal = init_goal[4]
    global n_goal = 2

    frame_id = 0
    curr_t = 0
    global lane_id = 0
    global left_prob = 0
    global keep_prob = 0
    global right_prob = 0

    result_f = open("./us101_2_cluster_100/cluster10/test_"*string(case_id)*".txt", "a+")

    # global to = TimerOutput()
    reset_timer!()

    while frame_id < 40
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
            global traj, future_wy, reach = get_next_position(cum_time, waypoints, 0.3, s_goal)
            if !reach
                # println("trajvalid1")
                trajvalid1 = check_traj(cum_time, waypoints, 0.1, frame_id+1, s_goal, scenario, start_frame)
                # println("trajvalid2")
                trajvalid2 = check_traj(cum_time, waypoints, 0.2, frame_id+2, s_goal, scenario, start_frame)
                # println("trajvalid3")
                trajvalid3 = check_traj(cum_time, waypoints, 0.3, frame_id+3, s_goal, scenario, start_frame)
                # println("trajvalid4")
                # trajvalid4 = check_traj(cum_time, waypoints, 0.4, frame_id+4, s_goal, scenario, start_frame)
                # trajvalid5 = check_traj(cum_time, waypoints, 0.5, frame_id+5, s_goal, scenario, start_frame)
                # trajvalid6 = check_traj(cum_time, waypoints, 0.6, frame_id+6, s_goal, scenario, start_frame)
                # trajvalid7 = check_traj(cum_time, waypoints, 0.7, frame_id+7, s_goal, scenario, start_frame)
                # trajvalid8 = check_traj(cum_time, waypoints, 0.8, frame_id+8, s_goal, scenario, start_frame)
                #trajvalid9 = check_traj(cum_time, waypoints, 0.9, frame_id+9, s_goal, scenario, start_frame)
                #trajvalid10 = check_traj(cum_time, waypoints, 1.0, frame_id+10, s_goal, scenario, start_frame)
                if !trajvalid1
                    println("traj1 invalid")
                end
                if !trajvalid2
                    println("traj2 invalid")
                end
                if !trajvalid3
                    println("traj3 invalid")
                end
                # if !trajvalid4
                #     println("traj4 invalid")
                # end
                # if !trajvalid5
                #     println("traj5 invalid")
                # end
                # if !trajvalid6
                #     println("traj6 invalid")
                # end
                # if !trajvalid7
                #     println("traj7 invalid")
                # end
                # if !trajvalid8
                #     println("traj8 invalid")
                # end
                #if !trajvalid9
                #    println("traj9 invalid")
                #end
                #if !trajvalid10
                #    println("traj10 invalid")
                #end
                # if !(trajvalid1 && trajvalid2&& trajvalid3)
                #    println("collision!!!!!")
                #    exit()
                # end
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
                break
            end
            global s_init = traj[end]
            print("init:", s_init)
            print("goal:", s_goal)
            println("goal distance:", sqrt((s_init[1]-s_goal[1])^2+(s_init[2]-s_goal[2])^2))
            if sqrt((s_init[1]-s_goal[1])^2+(s_init[2]-s_goal[2])^2+(s_init[3]-s_goal[3])^2+(s_init[4]-s_goal[4])^2) < 10
                println("congratulations!!!successful！！！")
                break
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
            if isValid(W, future_wy) && !new_start && length(future_wy) > 10 && false
                global cum_time = cal_time(future_wy)
                global waypoints = future_wy
                global M = zeros(4, length(waypoints))
                for i in 1:length(waypoints)
                    global M[:, i] = waypoints[i]
                end
                clf()
                show(W)
                scatter(s_init[1], s_init[2], c=:blue, s=50)
                scatter(s_goal[1], s_goal[2], c=:blue, s=50)
                plot(M[1, :], M[2, :], c=:blue, linewidth=1.5)
                savefig("./fig/test-cvae-real-data3-2-3-with-pre/real_data-"*string(frame_id)*".png")
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
                frame_id += 3
                continue
            end
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

        lane1_v = 0
        lane2_v = 0
        lane3_v = 0
        lane4_v = 0
        lane5_v = 0
        for v in neighbors1
            next_lane_id = predict_vehicle(scenario, center, convert(Int64, v[1]), frame_id+5, start_frame)
            if next_lane_id == 1
                lane1_v += 1
            elseif next_lane_id == 2
                lane2_v += 1
            elseif next_lane_id == 3
                lane3_v += 1
            elseif next_lane_id == 4
            lane4_v += 1
            elseif next_lane_id == 5
                lane5_v += 1
            end
        end
        if lane_id == 1
            global keep_prob, right_prob = softmax(1/lane1_v, 1/lane2_v)
            global left_prob = 0
        elseif lane_id == 2
            global left_prob, keep_prob, right_prob = softmax(1/lane1_v, 1/lane2_v, 1/lane3_v)
        elseif lane_id == 3
            global left_prob, keep_prob, right_prob = softmax(1/lane2_v, 1/lane3_v, 1/lane4_v)
        elseif lane_id == 4
            global left_prob, keep_prob, right_prob = softmax(1/lane3_v, 1/lane4_v, 1/lane5_v)
        elseif lane_id == 5
            global left_prob, keep_prob = softmax(1/lane4_v, 1/lane5_v)
            global right_prob = 0
        end

        # obstacle_set = set_obstacles(neighbors)
        if left_prob == 0
            left_prob = 0.0
        end
        if keep_prob == 0
            keep_prob = 0.0
        end
        if right_prob == 0
            right_prob = 0.0
        end
        # println(left_prob)
        # println(keep_prob)
        # println(right_prob)
        x_min, x_max, v_min, v_max = get_boundary(neighbors1, center, lane_id, s_goal)
        W = World(x_min, x_max, v_min, v_max, obstacle_set)
        # show(W)
        # scatter(s_init[1], s_init[2])
        # scatter(s_goal[1], s_goal[2])
        # savefig("./us101_2_cluster_100/cluster10/test_0_world.png")
        # exit()
        prob = [left_prob, keep_prob, right_prob]

        # println("goal:", s_goal)

        Nsample = 1000
        # right_prob = 1/3
        # left_prob = 1/3
        # keep_prob = 1/3
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
        # println(fmt.itr)
        # println(t)
        savefig("./us101_2_cluster_100/cluster10/test_case_"*string(case_id)*"_"*string(frame_id)*".png")
        frame_id += 3
        # break
        # print_timer()
    end
    open("./us101_2_cluster_100/cluster10/mytime"*string(case_id)*".txt","a+") do io
        print_timer(io, TimerOutputs.get_defaulttimer(), allocations=false)
    end
end
exit()
