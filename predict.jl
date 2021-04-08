using LinearAlgebra
using StaticArrays
using Distributions
using PyPlot
const Vec2f = SVector{2, Float64}
const Vec4f = SVector{4, Float64}
const VecFrame = SVector{18, Float64}

include("read_scenario.jl")

@inline function distance(p::Vec2f, q::Vec2f)
    return sqrt((p[1]-q[1])^2+(p[2]-q[2])^2)
end

function predict_vehicle(scenario::Vector{VecFrame}, center::Vec2f, vehicle_id::Int64, frame_id::Int64, start_frame)
    frame = get_frame(scenario, frame_id, start_frame)
    n = length(frame)
    next_lane_id = 0
    for i = 1:n
        if convert(Int64, frame[i][1]) == vehicle_id
            if distance(Vec2f(frame[i][5], frame[i][6]), center) < 150
                next_lane_id = frame[i][14]
                break
            end
        end
    end
    return next_lane_id
end

function cal_time(p::Vec4f, q::Vec4f)
    v = (sqrt(p[3]^2 + p[4]^2) + sqrt(q[3]^2 + q[4]^2)) / 2
    d = distance(Vec2f(p[1], p[2]), Vec2f(q[1], q[2]))
    return d/v
end

function cal_time(waypoints::Vector{Vec4f})
    n = length(waypoints)
    cum_time = [0.0]
    t = 0.0
    for i = 1:(n-1)
        t += cal_time(waypoints[i], waypoints[i+1])
        push!(cum_time, t)
    end
    return cum_time
end

function get_next_position(cum_time::Vector{Float64}, waypoints::Vector{Vec4f}, next_time_stamp::Float64)
    n = length(waypoints)
    t = 0
    for i = 1:(n-1)
        if cum_time[i] < next_time_stamp < cum_time[i+1]
            t = i + 1
            break
        end
    end
    return waypoints[1:t]
end

function get_next_position(cum_time::Vector{Float64}, waypoints::Vector{Vec4f}, next_time_stamp::Float64)
    n = length(waypoints)
    t = 0
    for i = 1:(n-1)
        if cum_time[i] < next_time_stamp < cum_time[i+1]
            t = i
            break
        end
    end
    println("the final time is ", cum_time[t])
    return waypoints[1:t], waypoints[t:end]
end

function get_next_position(cum_time::Vector{Float64}, waypoints::Vector{Vec4f}, next_time_stamp::Float64, s_goal::Vec4f)
    n = length(waypoints)
    t = 0
    reach = false
    println(length(waypoints))
    println(length(cum_time))
    for i = 1:(n-1)
        if cum_time[i] < next_time_stamp
            if distance(Vec2f(waypoints[i][1], waypoints[i][2]), Vec2f(s_goal[1], s_goal[2])) < 10
                t = i
                reach = true
                println("the final time is ", cum_time[t])
                break
            end
        end
        if cum_time[i] < next_time_stamp < cum_time[i+1]
            t = i
            break
        end
    end
    print(t)
    return waypoints[1:t], waypoints[t:end], reach
end


function get_traj(cum_time::Vector{Float64}, waypoints::Vector{Vec4f}, cur_time_stamp::Float64, next_time_stamp::Float64)
    n = length(waypoints)
    t = 0
    next_t = 0
    if cur_time_stamp - 0 < 1e-5
        t = 1
    end
    for i = 1:(n-1)
        if cum_time[i] < cur_time_stamp < cum_time[i+1]
            t = i
        end
        if cum_time[i] < next_time_stamp < cum_time[i+1]
            next_t = i
        end
    end
    return waypoints[t:next_t]
end

function check_traj(cum_time::Vector{Float64}, waypoints, time, frame_id, s_goal, scenario, start_frame)
    lane12 = (lane1_c + lane2_c)/2
    lane23 = (lane2_c + lane3_c)/2
    lane34 = (lane3_c + lane4_c)/2
    lane45 = (lane4_c + lane5_c)/2
    lane5edge = lane5_c + (lane5_c - lane45)
    traj = get_traj(cum_time, waypoints, time-0.1, time)
    # println(traj)
    s_init = traj[1]
    frame1 = get_frame(scenario, frame_id, start_frame)

    center = Vec2f(s_init[1], s_init[2])
    if 0 < s_init[1] < lane12
        lane_id = 1
    elseif lane12 < s_init[1] < lane23
        lane_id = 2
    elseif lane23 < s_init[1] < lane34
        lane_id = 3
    elseif lane34 < s_init[1] < lane45
        lane_id = 4
    elseif lane45 < s_init[1] < lane5edge
        lane_id = 5
    end
    neighbors1, front_neighbors1 = find_neighbors(frame1, center, lane_id, 150.0)
    obstacle_set = set_obstacles(neighbors1)
    x_min, x_max, v_min, v_max = get_boundary(neighbors1, center, lane_id, s_goal)
    W = World(x_min, x_max, v_min, v_max, obstacle_set)
    if isValid(W, traj)
        return true
    else
        return false
    end
end


function softmax(l1::Float64, l2::Float64, l3::Float64)
    sum = exp(l1) + exp(l2) + exp(l3)
    return convert(Float64, exp(l1)/sum), convert(Float64, exp(l2)/sum), convert(Float64, exp(l3)/sum)
end

function softmax(l1::Float64, l2::Float64)
    sum = exp(l1) + exp(l2)
    return convert(Float64, exp(l1)/sum), convert(Float64, exp(l2)/sum)
end
