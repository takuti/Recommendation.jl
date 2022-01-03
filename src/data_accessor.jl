export DataAccessor
export create_matrix, set_user_attribute, get_user_attribute, set_item_attribute, get_item_attribute

struct DataAccessor
    events::Array{Event,1}
    R::AbstractMatrix
    user_attributes::Dict{Int,Any} # user => attributes e.g. vector
    item_attributes::Dict{Int,Any} # item => attributes

    function DataAccessor(events::Array{Event,1}, n_user::Integer, n_item::Integer)
        R = create_matrix(events, n_user, n_item)
        new(events, R, Dict(), Dict())
    end

    function DataAccessor(R::AbstractMatrix)
        n_user, n_item = size(R)
        events = Array{Event,1}()

        for user in 1:n_user
            for item in 1:n_item
                r = R[user, item]
                if isa(r, Unknown)
                    R[user, item] = 0.0
                elseif !iszero(r)
                    push!(events, Event(user, item, r))
                end
            end
        end

        # cast to float matrix if an element type is integer; this is required by the following
        # manipulations relying on copy(R), which possibly updates the matrix values to
        # floating point numbers
        if Int <: eltype(R)
            R = map(Float64, R) # safe operation as all unknown values are already filled by 0
        end

        new(events, R, Dict(), Dict())
    end
end

function create_matrix(events::Array{Event,1}, n_user::Integer, n_item::Integer)
    R = zeros(n_user, n_item)
    for event in events
        # accumulate for implicit feedback events
        R[event.user, event.item] += event.value
    end
    R
end

function set_user_attribute(data::DataAccessor, user::Integer, attribute::AbstractVector)
    data.user_attributes[user] = attribute
end

function get_user_attribute(data::DataAccessor, user::Integer)
    get(data.user_attributes, user, [])
end

function set_item_attribute(data::DataAccessor, item::Integer, attribute::AbstractVector)
    data.item_attributes[item] = attribute
end

function get_item_attribute(data::DataAccessor, item::Integer)
    get(data.item_attributes, item, [])
end
