"""
Point d'entrée du programme. Lance le processus de contrôle du robot et l'interface de visualisation en temps réel.
"""

import multiprocessing as mp

from routines.routine1 import Routine1

from ui.dashboard import RealTimeApp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    data_queue = mp.Queue(maxsize=100)
    stop_event = mp.Event()

    routine = Routine1()

    process_run_robot = mp.Process(target=routine.run(), args=(data_queue, stop_event))
    process_run_robot.start()

    try:
        app = RealTimeApp(data_queue)
        app.run()
    finally:
        stop_event.set()
        process_run_robot.join()
        print("Programme arrêté proprement.")
