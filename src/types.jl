export Event

mutable struct Event
    user::Integer
    item::Integer
    value::Union{AbstractFloat, Integer} # e.g. rating, 0/1,
end
