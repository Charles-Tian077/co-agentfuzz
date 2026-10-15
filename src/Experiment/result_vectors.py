"""Parse recorded binary result vectors without executing their contents."""
from ast import literal_eval


def parse_result_vector(value):
    """Reject malformed observations rather than inventing missing outcomes."""
    vector = literal_eval(value) if isinstance(value, str) else value
    if not isinstance(vector, (list, tuple)):
        raise ValueError('Expected a list of binary results')
    if any(type(item) not in (int, bool) or item not in (0, 1) for item in vector):
        raise ValueError('Result observations must be binary')
    return [int(item) for item in vector]
