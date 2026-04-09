import socket
import struct
import time
import math

# ============================================================================
# CONFIGURATION DU SERVEUR
# ============================================================================
# MODE = "sinus"  # Options: "sinus" ou "fixed"
MODE = "fixed"
# ============================================================================

class UDPServer:
    def __init__(self, port):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)
        
        # Formats struct
        self.fmt_recv = "ddddd"   # 5 doubles reçus (40 octets)
        self.fmt_send = "8d"      # 8 doubles envoyés (64 octets)
        
        self.size_recv = struct.calcsize(self.fmt_recv)
        self.t = 0  # Compteur de temps pour les sinus
        self.first_packet_time = None  # Timestamp du premier packet
        
        print(f"Serveur actif sur le port {self.port}")
        print(f"Réception attendue : {self.size_recv} octets")
        print(f"Réponse prévue : {struct.calcsize(self.fmt_send)} octets")
        print(f"Mode: {MODE}")

    def run(self):
        try:
            while True:
                try:
                    # 1. Tentative de réception
                    data, addr = self.sock.recvfrom(1024)
                    
                    if len(data) == self.size_recv:
                        # Décodage des 5 valeurs reçues
                        v0, v1, v2, v3, grip = struct.unpack(self.fmt_recv, data)
                        
                        if MODE == "fixed":
                            # Mode positions fixes et vitesses à zéro
                            response_values = (0.0, math.pi/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        else:
                            # Mode sinus/cosinus
                            # Capturer le timestamp du premier packet
                            if self.first_packet_time is None:
                                self.first_packet_time = time.time()
                            
                            # Vérifier si 2 secondes se sont écoulées
                            elapsed = time.time() - self.first_packet_time
                            
                            if elapsed < 2.0:
                                # Pendant 2 secondes, renvoyer des zéros
                                response_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                            else:
                                # Après 2 secondes, générer les sinus/cosinus
                                self.t += 0.01
                                phases = [0, math.pi/2, math.pi] #, 3*math.pi/2]
                                
                                # Positions (sinus avec différentes phases)
                                positions = [math.pi/4] + [math.sin(self.t + phase) for phase in phases]
                                
                                # Vitesses (dérivées, cosinus avec mêmes phases)
                                velocities = [math.pi/4] + [math.cos(self.t + phase) for phase in phases]
                                
                                response_values = tuple(positions + velocities)
                        
                        # 2. Préparation de la réponse (64 octets)
                        response_bytes = struct.pack(self.fmt_send, *response_values)
                        
                        # 3. Envoi immédiat au client
                        self.sock.sendto(response_bytes, addr)
                        
                        print(f"Reçu 5 valeurs | Renvoyé 8 valeurs à {addr}")

                except BlockingIOError:
                    # On attend un peu pour ne pas saturer le CPU
                    time.sleep(0.005) 
                    continue
                except Exception as e:
                    print(f"Erreur : {e}")

        except KeyboardInterrupt:
            print("\nArrêt du serveur.")
        finally:
            self.sock.close()

if __name__ == "__main__":
    server = UDPServer(5005)
    server.run()