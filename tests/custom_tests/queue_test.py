from queue import Queue

q = Queue()
robot_pos = [0, 0, 0]  # Un objet mutable (une liste)

q.put(robot_pos)  # On met la référence dans la queue

robot_pos[0] = 90  # ON MODIFIE L'OBJET ORIGINAL
print(f"Original: {robot_pos}")

recup = q.get()  # On récupère l'objet de la queue
print(f"Récupéré: {recup}")
# Affiche [90, 0, 0] -> La modification a eu lieu !
