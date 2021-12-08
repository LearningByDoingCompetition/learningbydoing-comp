"""
Only change this file, if you know you have to and what you are doing.
Implement your controller in controller.py instead.
"""

from controller import controller
import numpy as np
import zmq

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("ipc://socket")

    while True:
        query = socket.recv_json()
        if query['init']:
            dim_ctrl = query['d_control']
            ctrl = controller(query['system'], dim_ctrl)

        nxtctrl = ctrl.get_input(
            np.asarray(query['state']).reshape(-1, 1),
            np.asarray(query['position']).reshape(-1, 1),
            np.asarray(query['target']).reshape(-1, 1)
        )

        socket.send_json(nxtctrl.flatten().tolist())
