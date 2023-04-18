class MockBtServer:

    def __init__(self):
        self.is_connected = False

    def Connect(self):
        self.is_connected = True
        return 0

    def Close(self):
        self.is_connected = False
