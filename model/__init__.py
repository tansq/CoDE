from .AC_CoDE import AC_CoDE

def select_model(model_name: str) -> object:
    if model_name == 'AC_CoDE':
        model = AC_CoDE(encoder='efficientnet-b4', pretrained=True)
    else:
        assert 'Unknown Model!'
    return model
