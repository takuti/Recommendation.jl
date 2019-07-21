export Event, States

# recommenders' states e.g. `is_built`
const States = Dict{Symbol,Any}

mutable struct Event
    user::Int
    item::Int
    value::Float64 # e.g. rating, 0/1
end
