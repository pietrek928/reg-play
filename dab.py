from model import OutputValue, InputValue, Model


class DABLowRef(Model):
    # Outputs
    VD = OutputValue(shape=(2,), descr='Both sides diode avg voltage')
    I = OutputValue(shape=(2,), descr='Both sides avg current')

    # Inputs
    V = InputValue(shape=(2,), descr='Both sides voltage')
    f = InputValue(desct='Switching frequency[Hz]')
    d = InputValue(descr='Switching phase shift')

    # Parameters
