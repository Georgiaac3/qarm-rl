# Physics-Based Simulator (Type Gazebo)

## 🎯 Objectif

Créer un **simulateur réaliste type Gazebo** avec:
- ⏰ **Horloge de simulation propre** (pas juste des steps)
- 🔄 **Physique intégrée** (dynamiques moteur, friction)
- 📊 **Boucle temps asynchrone** (physics @1kHz vs control @50Hz)
- 🤖 **Comportement réaliste** pour tester algorithmes complexes

---

## 📁 Fichiers

### 1. **`simulator_2d.py`** - Moteur de physique

```python
# Simulateur bas niveau avec:
Arm2DSimulator
  ├─ Dynamics integration (Euler method)
  ├─ Motor inertia & damping
  ├─ Friction (Coulomb + viscous)
  ├─ Joint limits with hard stops
  └─ Simulation clock

SimulationLoop
  ├─ High-frequency physics (1 kHz)
  ├─ Lower-frequency control (50 Hz)
  └─ Buffered command handling
```

### 2. **`env_2d_physics.py`** - Environnement Gymnasium

Compatible avec Gymnasium et Stable-Baselines3:

```python
Arm2DEnvPhysics(gym.Env)
  ├─ .reset() → obs, info
  ├─ .step(action) → obs, reward, terminated, truncated, info
  ├─ Observation: [target_x, target_y, q1, q2, q_dot1, q_dot2]
  ├─ Action: [acceleration_1, acceleration_2]
  └─ Utilise Arm2DSimulator internement
```

### 3. **`test_physics_simulator.py`** - Tests

```bash
python test_physics_simulator.py
```

Teste:
- Comparaison Simple env vs Physics env
- Trajectoires réalistes
- Génère `physics_simulator_test.png`

---

## 🔑 Différences clés

| Aspect | Simple Env | Physics Env |
|--------|-----------|------------|
| **Temps** | Steps discrets | Horloge continue |
| **Dynamique** | Position directe | Accélération → Vitesse → Position |
| **Friction** | Aucune | Coulomb + Visqueuse |
| **Inertie** | Aucune | Motor inertia réaliste |
| **Limits** | Hard clamp | Physical hard stops |
| **Ressemble à** | Tâche abstraite | Gazebo/Robot réel |

---

## 💻 Utilisation

### Utiliser le Physics Env au lieu du Simple Env

```python
# AVANT:
from env_2d import Arm2DEnv
env = Arm2DEnv()

# MAINTENANT:
from env_2d_physics import Arm2DEnvPhysics
env = Arm2DEnvPhysics()  # Même interface !
```

### Adapter train_2d.py

```python
# Dans train_2d.py, changer:
from env_2d import Arm2DEnv
# À:
from env_2d_physics import Arm2DEnvPhysics

def make_env():
    # env = Arm2DEnv(...)  # ANCIEN
    env = Arm2DEnvPhysics(...)  # NOUVEAU
    return env
```

### Adapter visualize_2d.py

Même changement d'import.

---

## 🧮 Physique Implémentée

### Dynamique Moteur

```
I * q_ddot = τ_motor - b*q_dot - f_friction

Où:
  I = motor_inertia (kg·m²)
  τ_motor = proportional à commande accélération
  b = damping (viscous coefficient)
  f_friction = friction coefficient × sign(q_dot)
```

### Intégration

Euler explicite (simple, stable pour small timesteps):

```
q_dot[t+1] = q_dot[t] + q_ddot[t] * dt_sim
q[t+1] = q[t] + q_dot[t+1] * dt_sim
```

### Fréquences

```
Physics Loop:
  dt_sim = 0.001 s (1 kHz)
  Intègre les équations du mouvement
  
Control Loop:
  dt_control = 0.05 s (50 Hz)
  Reçoit commandes du RL/IK
  Tourne ~50 cycles physiques par commande
```

---

## 🚀 Prochaines étapes

### 1. Tester Physics Env

```bash
python test_physics_simulator.py
```

Génère plot montrant trajectoires réalistes.

### 2. Re-entraîner RL sur Physics Env

```bash
# Créer train_physic_2d.py
# Identique à train_2d.py mais avec:
from env_2d_physics import Arm2DEnvPhysics

python train_physic_2d.py  # Entraîne sur Physics env
```

### 3. Comparer résultats

- IK seul sur Physics env
- IK + RL sur Physics env
- Voir si amélioration comparable

---

## 📊 Résultats Attendus

Avec Physics Env:
- Actions plus réalistes (accélérations vs velocités)
- Dynamics plus complexes → RL doit apprendre plus
- Bruit + friction → RL robustesse validation
- Transférable à robot réel

---

## 🔧 Paramètres Ajustables

### Arm2DSimulator

```python
Arm2DSimulator(
    l1=0.5,              # Segment length 1
    l2=0.5,              # Segment length 2
    dt_sim=0.001,        # Physics timestep (1 ms)
    dt_control=0.05,     # Control update (50 ms)
    motor_inertia=0.01,  # Joint inertia
    damping=0.1,         # Viscous damping
)
```

### Arm2DEnvPhysics

```python
Arm2DEnvPhysics(
    max_episode_duration=5.0,    # Max sim time
    motor_inertia=0.01,          # Make harder/easier
    damping=0.1,                 # More friction?
    target_radius=1.0,           # Workspace size
)
```

---

## 🐛 Débogage

Affiche infos de la simu:

```python
env = Arm2DEnvPhysics()
obs, info = env.reset()

for step in range(10):
    action = np.array([1.0, -0.5])
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Accès aux infos physiques:
    print(f"Sim time: {info['sim_time']}")     # Horloge
    print(f"q: {info['q']}")                    # Angles
    print(f"q_dot: {info['q_dot']}")            # Velocités
    print(f"distance: {info['distance']}")      # Target
```

---

## ✅ Checklist

- [ ] Test physics_simulator.py
- [ ] Créer train_physic_2d.py (adapté)
- [ ] Entraîner RL sur Physics env
- [ ] Comparer resultats vs simple env
- [ ] Adapter au QArm réel ensuite

---

**Prêt pour tester ? Exécute:**

```bash
python test_physics_simulator.py
```
