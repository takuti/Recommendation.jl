export Event, Parameters

typealias Parameters Dict{Symbol,Any}

type Event
    user::Int
    item::Int
    value::Float64 # e.g. rating, 0/1
end
