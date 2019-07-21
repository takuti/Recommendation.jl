export DataAccessor
export create_matrix, set_user_attribute, set_item_attribute

struct DataAccessor
    events::Array{Event,1}
    R::AbstractMatrix
    user_attributes::Dict{Int,Any} # user => attributes e.g. vector
    item_attributes::Dict{Int,Any} # item => attributes

    function DataAccessor(events::Array{Event,1}, n_user::Int, n_item::Int)
        R = create_matrix(events, n_user, n_item)
        new(events, R, Dict(), Dict())
    end

    function DataAccessor(R::AbstractMatrix)
        n_user, n_item = size(R)
        events = Array{Event,1}()

        for user in 1:n_user
            for item in 1:n_item
                r = R[user, item]
                if !isnan(r) && r != 0
                    append!(events, [Event(user, item, r)])
                end
            end
        end

        new(events, R, Dict(), Dict())
    end
end

function create_matrix(events::Array{Event,1}, n_user::Int, n_item::Int)
    R = ones(n_user, n_item) * NaN
    for event in events
        if isnan(R[event.user, event.item])
            R[event.user, event.item] = 0
        end

        # accumulate for implicit feedback events
        R[event.user, event.item] += event.value
    end
    R
end

function set_user_attribute(data::DataAccessor, user::Int, attribute::AbstractVector)
    data.user_attributes[user] = attribute
end

function set_item_attribute(data::DataAccessor, item::Int, attribute::AbstractVector)
    data.item_attributes[item] = attribute
end
