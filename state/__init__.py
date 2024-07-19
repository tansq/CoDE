from .state import State

def select_state(state_type):
    # initialize state
    if state_type == 'default':
        state = State()
    else:
        assert 'unknown state.'
    return state