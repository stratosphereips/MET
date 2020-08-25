
class Base:

    _config = None
    _logger = None

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError("Attacks must implement run method!")