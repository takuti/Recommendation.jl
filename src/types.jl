export Unknown, Infinite, Event

Unknown = Union{Missing, Nothing}
Infinite = Union{AbstractFloat, Integer}

mutable struct Event
    user::Integer
    item::Integer
    value::Infinite # e.g. rating, 0/1,
end
