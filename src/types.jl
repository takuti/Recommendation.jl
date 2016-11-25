export Event, Parameters, States

# model parameters
typealias Parameters Dict{Symbol,Any}

# recommenders' states e.g. `is_built`
typealias States Dict{Symbol,Any}

type Event
    user::Int
    item::Int
    value::Float64 # e.g. rating, 0/1
end
