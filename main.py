"""
Point d'entrée du programme. Lance le processus de contrôle du robot et l'interface de visualisation en temps réel.
"""

import multiprocessing as mp

from routines.routine1 import Routine1
from src.robot_control.ui.dashboard import RealTimeApp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    data_queue = mp.Queue(maxsize=100)
    stop_event = mp.Event()

    routine = Routine1(display_data_queue=data_queue, stop_event=stop_event)

    routine_process = mp.Process(target=routine.run)
    routine_process.start()

    try:
        app = RealTimeApp(data_queue)
        app.run()
    finally:
        stop_event.set()
        routine_process.join()
        print("Programme arrêté proprement.")
