export Event, DataValue

DataValue = Union{Nothing, Missing, AbstractFloat, Integer}

mutable struct Event
    user::Integer
    item::Integer
    value::Union{AbstractFloat, Integer} # e.g. rating, 0/1,
end
