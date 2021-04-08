@inbounds @inline dist2(p, q)::Float64 = sqrt((p[1]-q[1])^2+(p[2]-q[2])^2)
# FMTree class
mutable struct FMTree
    s_init::Vec4f
    s_goal::Vec4f
    N #number of samples
    Pset::Vector{Vec4f} # Point set
    cost::Vector{Float64} #cost
    time::Vector{Float64} #optimal time to connect one node to its parent node
    parent::Vector{Int64} #parent node
    bool_unvisit::BitVector #logical value for Vunvisit
    bool_open::BitVector #logical value for Open
    bool_closed::BitVector #logical value for Closed
    world::World # simulation world config
    itr::Int64 # iteration num
    reach_goal_region::Bool
    end_index::Int64
    idx::Int64
    cannot_find_open_point::Bool

    function FMTree(s_init::Vec4f, s_goal::Vec4f, N, world, right_prob::Float64, left_prob::Float64, keep_prob::Float64, lane_id::Int64, frames)
        # constructer: sampling valid point from the configurationspace
        println("initializing fmt ...")
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
        reach_goal_region = false
        Pset = Vec4f[]
        push!(Pset, s_init) #inply idx_init = 1
        myrn(min, max) = min + (max-min)*rand()
        gaussianrn(mean, var) = rand(Normal(mean, var), 1)[1]
        distance(x0, x1) = sqrt((x1[2] - x0[2])^2 + (x1[1] - x0[1])^2)
        uniform_percent = 0.0
        N = N - 2
        N_uniform = Int(round(Int(round(N * uniform_percent))/5))
        N_prob = N - (N_uniform*5)
        N = N + 2

        #=
        f_r = open("test_host_file3.csv", "r")
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
        =#

        # n_scenario = length(scenario)-1
        # sample_each_scenario = convert(Int64, floor(N_prob/n_scenario))
        # rest_point = N - 2 - N_uniform - sample_each_scenario*n_scenario
        for n = 1:N_uniform
            while(true)
                x = myrn(world.x_min[1], world.x_max[1])
                y = myrn(s_init[2], world.x_max[2])
                p = Vec4f(x, y,
                        myrn(world.v_min[1], world.v_max[1]),
                        myrn(world.v_min[2], world.v_max[2]))
                if isValid(world, p)
                    push!(Pset, p)
                    i = 1
                    while i < 5
                        p = Vec4f(x, y,
                            myrn(world.v_min[1], world.v_max[1]),
                            myrn(world.v_min[2], world.v_max[2]))
                        if isValid(world, p)
                            push!(Pset, p)
                            i += 1
                        end
                    end
                    break
                end
            end
        end
        theta = atan(s_init[4]/s_init[3]) + pi
        slope = s_init[4]/s_init[3]
        b = s_init[2] - slope * s_init[1]
        myfun(slope, b, x) = slope * x + b
        if slope < 0
            if lane_id == 1
                x_min = 0
            else
                left_lane = lane_c[lane_id - 1]
                x_min = left_lane
            end
            x_max = s_init[1]
        else
            if lane_id == 5
                x_max = lane5edge
            else
                right_lane = lane_c[lane_id + 1]
                x_max = right_lane
            end
            x_min = s_init[1]
        end
        for n = 1:N_prob
            # println(n)
            while true
                global mean_x = myrn(x_min, x_max)
                global mean_y = myfun(slope, b, mean_x)
                p = Vec4f(mean_x, mean_y, 1.0, 1.0)
                if isValid(world, p)
                    break
                end
            end
            # println(mean_x)
            # println(mean_y)
            while true
                p_x = gaussianrn(mean_x, 2)
                p_y = gaussianrn(mean_y, 2)
                if distance([p_x, p_y], s_init) < distance([p_x, p_y], s_goal)
                    x_v = gaussianrn(s_init[3], 3)
                    y_v = gaussianrn(s_init[4], 3)
                else
                    x_v = gaussianrn(s_goal[3], 3)
                    y_v = gaussianrn(s_goal[4], 3)
                end
                p = Vec4f(p_x, p_y, x_v, y_v)
                if isValid(world, p)
                    push!(Pset, p)
                    break
                end
            end
        end


        push!(Pset, s_goal) #inply idx_goal = N [last]
        println(N)
        cost = zeros(N)
        time = zeros(N)
        parent = ones(Int, N)
        bool_unvisit = trues(N)
        bool_unvisit[1] = false
        bool_closed = falses(N)
        bool_open = falses(N)
        bool_open[1] = true
        println("finish initializing")
        new(s_init, s_goal,
            N, Pset, cost, time, parent, bool_unvisit, bool_open, bool_closed, world, 0, false, 0, 0, false)
    end
end

function show(this::FMTree)
    println("drawing...")
    show(this.world)
    N = length(this.Pset)
    mat = zeros(2, N)
    for idx = 1:N
        mat[:, idx] = this.Pset[idx][1:2]
    end
    idxset_open = findall(this.bool_open)
    idxset_closed = findall(this.bool_closed)
    idxset_unvisit = findall(this.bool_unvisit)
    idxset_tree = setdiff(union(idxset_open, idxset_closed), [1])
    scatter(mat[1, 1], mat[2, 1], c=:blue, s=10, zorder = 100)
    scatter(mat[1, end], mat[2, end], c=:blue, s=10, zorder = 101)
    scatter(mat[1, idxset_open], mat[2, idxset_open], c=:gray, s=2)
    scatter(mat[1, idxset_closed], mat[2, idxset_closed], c=:gray, s=2)
    # println(size(mat))
    scatter(mat[1, idxset_unvisit], mat[2, idxset_unvisit], c=:orange, s=5)
    for idx in idxset_tree
        s0 = this.Pset[this.parent[idx]]
        s1 = this.Pset[idx]
        tau = this.time[idx]
        show_trajectory(s0, s1, tau)
    end

    scatter(mat[1, 1], mat[2, 1], c=:blue, s=20, zorder = 100)
    scatter(mat[1, end], mat[2, end], c=:blue, s=20, zorder = 101)
    println("s_init in show funciton:", this.s_init)
    scatter(this.s_init[1], this.s_init[2], c=:red, s=50)
    scatter(this.s_goal[1], this.s_goal[2], c=:red, s=50)

    # xlim(this.world.x_min[1]-5, this.world.x_max[1]+5)
    # ylim(this.world.x_min[2]-5, this.world.x_max[2]+5)
    println("finish drawing")
end

function solve(this::FMTree, frame_id::Int64, with_savefig = false)
    # keep extending the node until the tree reaches the goal
    println("please set with_savefig=false if you want to measure the computation time" )
    println("start solving")
    while(true)
        extend(this)
        if with_savefig
            if ((this.itr<100) & (this.itr % 20 == 1)) || (this.itr % 200==1)
                close()
                show(this)
                savefig("./fig/test-uniform-without-pre-real-data3/"*string(frame_id)*"-"*string(this.itr)*".png")
            end
        end
        if this.itr > 500 || this.cannot_find_open_point
            opt_idx = 1
            open_set = vcat(findall(this.bool_open), findall(this.bool_closed))
            min_dist = 1500
            for i in open_set
                x = this.Pset[i]
                dist = sqrt((this.s_goal[2] - x[2])^2 + (this.s_goal[1] - x[1])^2)
                if dist < min_dist
                    opt_idx = i
                    min_dist = dist
                end
            end
            if opt_idx != 1
                this.reach_goal_region = true
                this.end_index = opt_idx
            else
                return Int64[]
            end
        end
        if !this.bool_unvisit[end]
            this.idx = this.N
            println("finish normally")
            break
        elseif this.reach_goal_region
            this.idx = this.end_index
            println("finish the goal region")
            break
        end
    end
    idx_solution = Int64[this.idx]
    while(true)
        println("idx:", this.idx)
        this.idx = this.parent[this.idx]
        push!(idx_solution, this.idx)
        this.idx == 1 && break
    end
    println("finish solving")
    return idx_solution
end

function extend(this::FMTree)
    # extend node
    this.itr += 1
    r = 200.0
    # if this.itr == 3
    #     exit()
    # end
    # println(this.itr)
    idxset_open = findall(this.bool_open)
    idxset_unvisit = findall(this.bool_unvisit)

    # println(this.cost)
    this.cannot_find_open_point = false
    try
        idx_lowest = idxset_open[findmin(this.cost[idxset_open])[2]]
    catch ArgumentError
        println("cannot find open point")
        this.cannot_find_open_point = true
        return
    end
    # println(idxset_unvisit)
    idx_lowest = idxset_open[findmin(this.cost[idxset_open])[2]]
    idxset_near, _, _ = filter_reachable(this.Pset, idxset_unvisit,
                                      this.Pset[idx_lowest], r, :F)

    for idx_near in idxset_near
        idxset_cand, distset_cand, timeset_cand = filter_reachable(this.Pset, idxset_open,
                                                    this.Pset[idx_near], r, :B)
        isempty(idxset_cand) && return
        cost_new, idx_costmin = findmin(this.cost[idxset_cand] + distset_cand)
        time_new = timeset_cand[idx_costmin] # optimal time for new connection
        idx_parent = idxset_cand[idx_costmin]
        waypoints = gen_trajectory(this.Pset[idx_parent], this.Pset[idx_near], time_new, 50)
        if isValid(this.world, waypoints)
            this.bool_unvisit[idx_near] = false
            this.bool_open[idx_near] = true
            this.cost[idx_near] = cost_new
            this.time[idx_near] = time_new
            this.parent[idx_near] = idx_parent
            x0 = this.Pset[idx_near]
            # println("goal region:", sqrt((this.s_goal[2] - x0[2])^2 + (this.s_goal[1] - x0[1])^2
            #        + (this.s_goal[3] - x0[3])^2 + (this.s_goal[4] - x0[4])^2))
            if sqrt((this.s_goal[2] - x0[2])^2 + (this.s_goal[1] - x0[1])^2 +
                (this.s_goal[3] - x0[3])^2 + (this.s_goal[4] - x0[4])^2) < 5
                this.reach_goal_region = true
                this.end_index = idx_near
                println("goal region:", sqrt((this.s_goal[2] - x0[2])^2 + (this.s_goal[1] - x0[1])^2
                        + (this.s_goal[3] - x0[3])^2 + (this.s_goal[4] - x0[4])^2))
                break
            end
        end
    end
    this.bool_open[idx_lowest] = false
    this.bool_closed[idx_lowest] = true
end
