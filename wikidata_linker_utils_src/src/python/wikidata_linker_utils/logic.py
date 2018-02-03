from numpy import logical_and, logical_not, logical_or


def logical_negate(truth, falses):
    out = truth
    for value in falses:
        out = logical_and(out, logical_not(value))
    return out


def logical_ors(values):
    assert(len(values) > 0), "values cannot be empty."
    out = values[0]
    for val in values[1:]:
        out = logical_or(out, val)
    return out


def logical_ands(values):
    assert(len(values) > 0), "values cannot be empty."
    out = values[0]
    for val in values[1:]:
        out = logical_and(out, val)
    return out
