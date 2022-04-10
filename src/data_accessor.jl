export DataAccessor
export create_matrix, set_user_attribute, get_user_attribute, set_item_attribute, get_item_attribute, split_data

struct DataAccessor
    events::Array{Event,1}
    R::AbstractMatrix
    user_attributes::Dict{Int,Any} # user => attributes e.g. vector
    item_attributes::Dict{Int,Any} # item => attributes

    function DataAccessor(events::Array{Event,1}, n_users::Integer, n_items::Integer)
        R = create_matrix(events, n_users, n_items)
        new(events, R, Dict(), Dict())
    end

    function DataAccessor(R::AbstractMatrix)
        n_users, n_items = size(R)
        events = Array{Event,1}()

        for user in 1:n_users
            for item in 1:n_items
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

function create_matrix(events::Array{Event,1}, n_users::Integer, n_items::Integer)
    R = zeros(n_users, n_items)
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

function split_data(data::DataAccessor, n_folds::Integer)
    if n_folds < 2
        error("`n_folds` must be greater than 1 to split the samples into train and test sets.")
    end

    events = shuffle(data.events)
    n_events = length(events)

    if n_folds > n_events
        error("`n_folds = $n_folds` must be less than $n_events, the number of all samples.")
    end

    n_users, n_items = size(data.R)

    step = convert(Integer, round(n_events / n_folds))

    if n_folds == n_events
        @info "Splitting $n_events samples for leave-one-out cross validation"
    else
        @info "Splitting $n_events samples for $n_folds-fold cross validation"
    end

    train_test_pairs = Array{Tuple{DataAccessor, DataAccessor},1}()

    for (index, head) in enumerate(1:step:n_events)
        tail = min(head + step - 1, n_events)

        truth_events = events[head:tail]
        truth_data = DataAccessor(truth_events, n_users, n_items)

        train_events = vcat(events[1:head - 1], events[tail + 1:end])
        train_data = DataAccessor(train_events, n_users, n_items)

        push!(train_test_pairs, (train_data, truth_data))

        @debug "fold#$index will test the samples in [$head, $tail]"
    end

    train_test_pairs
end
