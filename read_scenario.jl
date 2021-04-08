using LinearAlgebra
using StaticArrays
using Distributions
using PyPlot
const Vec2f = SVector{2, Float64}
const Vec4f = SVector{4, Float64}
const VecFrame = SVector{18, Float64}

include("src/geometry/geometry.jl")
include("src/world.jl")
include("src/double_integrator.jl")

@inline function distance(p::Vec2f, q::Vec2f)
    return sqrt((p[1]-q[1])^2+(p[2]-q[2])^2)
end

function read_scenario(file_name)
    f_r = open(file_name, "r")
    n = countlines(f_r)
    seekstart( f_r )

    scenario = VecFrame[]
    for i = 1:n
        line = split( readline( f_r ), "," )
        frame = VecFrame(parse(Float64, line[1] ),
                    parse(Float64, line[2] ),
                    parse(Float64, line[3] ),
                    parse(Float64, line[4] ),
                    parse(Float64, line[5] ),
                    parse(Float64, line[6] ),
                    parse(Float64, line[7] ),
                    parse(Float64, line[8] ),
                    parse(Float64, line[9] ),
                    parse(Float64, line[10] ),
                    parse(Float64, line[11] ),
                    parse(Float64, line[12] ),
                    parse(Float64, line[13] ),
                    parse(Float64, line[14] ),
                    parse(Float64, line[15] ),
                    parse(Float64, line[16] ),
                    parse(Float64, line[17] ),
                    parse(Float64, line[18] ))
        push!(scenario, frame)
    end
    start_frame=scenario[1][2]
    print(start_frame)
    return scenario, start_frame
end

function read_sample_points(file_name)
    f_r = open(file_name, "r")
    n = countlines(f_r)
    seekstart( f_r )

    sample_points = Vec4f[]
    for i = 1:n
        line = split( readline( f_r ), " " )
        p = Vec4f(parse(Float64, line[1] ),
                    parse(Float64, line[2] ),
                    parse(Float64, line[3] ),
                    parse(Float64, line[4] ))
        push!(sample_points, p)
    end
    return sample_points
end

function get_frame(scenario::Vector{VecFrame}, frame_num::Int64, start_frame)
    # start_frame = 1530
    choose_frame = start_frame + frame_num
    frame = VecFrame[]
    n = length(scenario)
    for i = 1:n
        if convert(Int64, scenario[i][2]) == choose_frame
            push!(frame, scenario[i])
        end
    end
    return frame
end

function find_neighbors(frame::Vector{VecFrame}, center::Vec2f, lane_id::Int64, r::Float64)
    n = length(frame)
    neighbors = VecFrame[]
    front_neighbors = VecFrame[]
    for i = 1:n
        v = Vec2f(frame[i][5], frame[i][6])
        if distance(v, center) < r
            push!(neighbors, frame[i])
            if v[2] >= center[2]
                push!(front_neighbors, frame[i])
            end
        end
    end
    return neighbors, front_neighbors
end

function set_obstacles(neighbors::Vector{VecFrame})
    n = length(neighbors)
    obstacle_set = Rectangle[]
    for i = 1:n
        obs = Rectangle([neighbors[i][5], neighbors[i][6]], neighbors[i][10], neighbors[i][9])
        push!(obstacle_set, obs)
    end
    return obstacle_set
end

function get_boundary(neighbors::Vector{VecFrame}, ego_vehicle::Vec2f, lane_id::Int64, s_goal::Vec4f)
    lane1_c = 7.231027419583393
    lane2_c = 18.84918099161813
    lane3_c = 29.990037979406054
    lane4_c = 40.99582664476257
    lane5_c = 53.099915981198585
    lane12 = (lane1_c + lane2_c)/2
    lane23 = (lane2_c + lane3_c)/2
    lane34 = (lane3_c + lane4_c)/2
    lane45 = (lane4_c + lane5_c)/2
    lane5edge = lane5_c + (lane5_c - lane45)
    center = ego_vehicle
    n = length(neighbors)
    y_max = neighbors[1][6]
    y_min = neighbors[1][6]
    for i = 1:n
        if neighbors[i][6] > y_max
            y_max = neighbors[i][6]
        end
        if neighbors[i][6] < y_min
            y_min = neighbors[i][6]
        end
    end
    if y_min > center[2]
        y_min = center[2]
    end
    if y_max < s_goal[2]
        y_max = s_goal[2] + 10
    end
    xmin = [0.0, 0.0]
    xmax = [0.0, 0.0]
    # println("ego vehicle:", center)
    if lane_id == 1
        global xmin = [0, y_min-10]
        global xmax = [lane23, y_max]
    elseif lane_id == 2
        global xmin = [0, y_min-10]
        global xmax = [lane34, y_max]
    elseif lane_id == 3
        global xmin = [lane12, y_min-10]
        global xmax = [lane45, y_max]
    elseif lane_id == 4
        global xmin = [lane23, y_min-10]
        global xmax = [lane5edge, y_max]
    elseif lane_id == 5
        global xmin = [lane34, y_min-10]
        global xmax = [lane5edge, y_max]
    end
    vmin = [-20.0, 0.0]
    vmax = [20.0, 100.0]
    return xmin, xmax, vmin, vmax
end

function nearest_front_vehicle(neighbors::Vector{VecFrame}, host::Vec2f, lane_id::Int64)
    min_d = 1500
    nearest_v = neighbors[1]
    for v in neighbors
        if v[14] == lane_id && v[6] > host[2]
            if distance(Vec2f(v[5], v[6]), host) < min_d
                min_d = distance(Vec2f(v[5], v[6]), host)
                nearest_v = v
            end
        end
    end
    return min_d, nearest_v[12]
end

function nearest_front_left_vehicle(neighbors::Vector{VecFrame}, host::Vec2f, lane_id::Int64)
    min_d = 1500
    nearest_v = neighbors[1]
    for v in neighbors
        if v[14] == lane_id-1 && v[6] > host[2]
            if 100 < distance(Vec2f(v[5], v[6]), host) < min_d
                min_d = distance(Vec2f(v[5], v[6]), host)
                nearest_v = v
                y_d = v[6] - host[2]
            end
        end
    end
    return y_d, nearest_v[12]
end

function nearest_front_right_vehicle(neighbors::Vector{VecFrame}, host::Vec2f, lane_id::Int64)
    min_d = 1500
    nearest_v = neighbors[1]
    y_d = 0.0
    for v in neighbors
        if v[14] == lane_id+1 && v[6] > host[2]
            if 100 < distance(Vec2f(v[5], v[6]), host) < min_d
                min_d = distance(Vec2f(v[5], v[6]), host)
                nearest_v = v
                y_d = v[6] - host[2]
            end
        end
    end
    return y_d, nearest_v[12]
end

function nearest_back_vehicle(neighbors::Vector{VecFrame}, host::Vec2f, lane_id::Int64)
    min_d = 1500
    nearest_v = neighbors[1]
    y_d = 0.0
    for v in neighbors
        if v[14] == lane_id && v[6] < host[2]
            if distance(Vec2f(v[5], v[6]), host) < min_d
                min_d = distance(Vec2f(v[5], v[6]), host)
                nearest_v = v
            end
        end
    end
    return min_d, nearest_v[12]
end


function get_init_goal(file_name)
    f_host = open(file_name, "r")
    n = countlines(f_host)
    seekstart( f_host )

    host_scenario = VecFrame[]
    for i = 1:n
        line = split( readline( f_host ), "," )
        frame = VecFrame(parse(Float64, line[1] ),
                    parse(Float64, line[2] ),
                    parse(Float64, line[3] ),
                    parse(Float64, line[4] ),
                    parse(Float64, line[5] ),
                    parse(Float64, line[6] ),
                    parse(Float64, line[7] ),
                    parse(Float64, line[8] ),
                    parse(Float64, line[9] ),
                    parse(Float64, line[10] ),
                    parse(Float64, line[11] ),
                    parse(Float64, line[12] ),
                    parse(Float64, line[13] ),
                    parse(Float64, line[14] ),
                    parse(Float64, line[15] ),
                    parse(Float64, line[16] ),
                    parse(Float64, line[17] ),
                    parse(Float64, line[18] ))
        push!(host_scenario, frame)
    end
    init_goal = Vec4f[]
    for i = 1:(n-1)
        cur = host_scenario[i]
        next = host_scenario[i+1]
        theta = atan((next[6]-cur[6])/(next[5]-cur[5]))
        if theta < 0
            theta = pi + theta
        end
        v_x = cur[12] * cos(theta)
        v_y = cur[12] * sin(theta)
        p = Vec4f(cur[5], cur[6], v_x, v_y)
        if i%10 == 1
            push!(init_goal, p)
        end
    end
    cur = host_scenario[n-1]
    next = host_scenario[n]
    theta = atan((next[6]-cur[6])/(next[5]-cur[5]))
    if theta < 0
        theta = pi + theta
    end
    v_x = next[12] * cos(theta)
    v_y = next[12] * sin(theta)
    p = Vec4f(next[5], next[6], v_x, v_y)
    push!(init_goal, p)

    return init_goal
end
