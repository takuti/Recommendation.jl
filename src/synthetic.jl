export Feature, Rule, accumulate, generate

struct Feature
    name::String
    default::Any
    random::Function
end

struct Rule
    # return bool
    f::Function

    # if f returns true, accumulative CTR is lifted p%
    p::Float64
end

function accumulate(sample::Dict, rules::Array{Rule,1})
    ctr = 0.0
    for rule in rules
        if rule.f(sample)
            ctr += rule.p
        end
    end
    ctr
end

function generate(sample::Dict, rules::Array{Rule,1})
    rand() <= accumulate(sample, rules)
end

function generate(samples::Array{Dict,1}, rules::Array{Rule,1})
    map(samples) do sample
        generate(sample, rules)
    end
end
