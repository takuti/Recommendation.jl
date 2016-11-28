export DataAccessor
export create_matrix, set_user_attribute, set_item_attribute

immutable DataAccessor
    events::Array{Event,1}
    R::AbstractMatrix
    user_attributes::Dict{Int,Any} # user => attributes e.g. vector
    item_attributes::Dict{Int,Any} # item => attributes
end

DataAccessor(events::Array{Event,1}, n_user::Int, n_item::Int) = begin
    R = create_matrix(events, n_user, n_item)
    DataAccessor(events, R, Dict(), Dict())
end

DataAccessor(R::AbstractMatrix) = begin
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

    DataAccessor(events, R, Dict(), Dict())
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

function set_user_attribute(da::DataAccessor, user::Int, attribute::AbstractVector)
    da.user_attributes[user] = attribute
end

function set_item_attribute(da::DataAccessor, item::Int, attribute::AbstractVector)
    da.item_attributes[item] = attribute
end
