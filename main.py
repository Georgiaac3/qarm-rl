"""
Point d'entrée du programme. Lance le processus de contrôle du robot et l'interface de visualisation en temps réel.
"""

import argparse
import multiprocessing as mp

from routines.routine1 import Routine1
from routines.routine2 import Routine2
from routines.routine3 import Routine3
from src.robot_control.ui.dashboard import RealTimeApp


def parse_args():
    parser = argparse.ArgumentParser(description="Lance une routine du robot QARM.")
    parser.add_argument(
        "--routine",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="Numéro de la routine à lancer (1 ou 2 ou 3).",
    )
    return parser.parse_args()


def build_routine(routine_id, data_queue, stop_event):
    if routine_id == 1:
        return Routine1(display_data_queue=data_queue, stop_event=stop_event)
    if routine_id == 2:
        return Routine2(display_data_queue=data_queue, stop_event=stop_event)
    if routine_id == 3:
        return Routine3(display_data_queue=data_queue, stop_event=stop_event)
    raise ValueError(f"Routine inconnue: {routine_id}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    data_queue = mp.Queue(maxsize=100)
    stop_event = mp.Event()

    routine = build_routine(args.routine, data_queue, stop_event)

    routine_process = mp.Process(target=routine.run)
    routine_process.start()

    try:
        app = RealTimeApp(data_queue)
        app.run()
    finally:
        stop_event.set()
        routine_process.join()
        print("Programme arrêté proprement.")
