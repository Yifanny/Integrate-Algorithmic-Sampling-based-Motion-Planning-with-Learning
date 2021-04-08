mutable struct World
    x_min
    x_max
    v_min
    v_max
    Pset::Vector{Polygonic}
    function World(x_min, x_max, v_min, v_max, Pset)
        new(x_min, x_max, v_min, v_max, Pset)
    end
end

@inline function isValid(this::World, s_q::Vec4f)
    # check if the sampled point is inside the world"
    # @inbounds !(this.x_min[1]<s_q[1]<this.x_max[1]) && return false
    # @inbounds !(this.x_min[2]<s_q[2]<this.x_max[2]) && return false
    if !(this.x_min[1]<s_q[1]<this.x_max[1])
        # println("x wrong ", this.x_min[1], " ", s_q[1], " ", this.x_max[1])
        return false
    end
    if !(this.x_min[2]<s_q[2]<this.x_max[2])
        # println("y wrong ", this.x_min[2], " ", s_q[2], " ", this.x_max[2])
        return false
    end
    # @inbounds !(this.v_min[1]<s_q[3]<this.v_max[1]) && return false
    # @inbounds !(this.v_min[2]<s_q[4]<this.v_max[2]) && return false
    if !(this.v_min[1]<s_q[3]<this.v_max[1])
        # println("v1 wrong ", s_q[3])
        return false
    end
    if !(this.v_min[2]<s_q[4]<this.v_max[2])
        # println("v2 wrong ", s_q[4])
        return false
    end
    for P in this.Pset
        if isInside(P, Vec2f(s_q[1:2]))
            # println("pos wrong ", s_q[1:2])
            return false
        end
    end
    return true
end

@inline function isValid(this::World, q_set::Vector{Vec4f})
    # check validity for multiple points.
    # will be used for piecewize path consited of multiple points
    for q in q_set
        !isValid(this, q) && return false
    end
    return true
end

@inline function isIntersect(this::World, q1::Vec4f, q2::Vec4f)
    for P in this.Pset
        isIntersect(P, q1, q2) && return true
    end
    return false
end


function show(this::World)
    p1 = [this.x_min[1], this.x_min[2]]
    p2 = [this.x_min[1], this.x_max[2]]
    p3 = [this.x_max[1], this.x_max[2]]
    p4 = [this.x_max[1], this.x_min[2]]
    # axis("scaled")
    plot([p1[1], p2[1]], [p1[2], p2[2]], "k-")
    plot([p2[1], p3[1]], [p2[2], p3[2]], "k-")
    plot([p3[1], p4[1]], [p3[2], p4[2]], "k-")
    plot([p4[1], p1[1]], [p4[2], p1[2]], "k-")
    for P in this.Pset
        show(P)
    end
end
