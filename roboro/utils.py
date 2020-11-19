def apply_to_state_list(func, state_list):
    """Apply function to whole list of states (e.g. concatenate, stack etc.)
    Recursively apply this function to nested dicts."""
    if isinstance(state_list[0], dict):
        return {
            key: apply_to_state_list(func, [state[key] for state in state_list])
            for key in state_list[0]
        }
    else:
        return func(state_list)


def apply_to_state(func, state):
    """Apply function recursively to state dict or directly to state"""
    if isinstance(state, dict):
        return apply_rec_to_dict(func, state)
    else:
        return func(state)


def apply_rec_to_dict(func, tensor_dict):
    """Apply a function recursively to every non dict object in a nested dict"""
    zipped = zip(tensor_dict.keys(), tensor_dict.values())
    return {
        key: apply_rec_to_dict(func, content) if isinstance(content, dict)
        else func(content)
        for key, content in zipped
    }
