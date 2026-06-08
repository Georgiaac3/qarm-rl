# QARM-RL

Système de contrôle du bras robotique QARM via communication UDP avec Simulink. Implémente une séquence de lancer d'objet avec contrôle proportionnel des vitesses articulaires.

## Structure du Projet

```
TODO
qarm-rl/
├── main.py                          # Point d'entrée principal
├── BasicIO_pwm_mode_retest.slx      # Modèle Simulink du QARM
├── requirements.txt                 # Dépendances Python
├── README.md                        # Documentation
│
├── core/                            # Modules centraux
│   ├── config.py                    # Configuration centralisée (Settings)
│   └── logger.py                    # Configuration du logging
│
└── utils/                           # Modules utilitaires
    ├── brain.py                     # Logique de contrôle (à venir)
    ├── camera.py                    # Traitement d'images (à venir)
    └── udp.py                       # Communication UDP avec Simulink
```

## Installation

### Prérequis

- Python 3.8 ou supérieur (idéalement 3.12, plus haut ça risque de faire bigger pyrealsense)
- MATLAB/Simulink (pour le modèle du robot)

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone <url-du-repo>
   cd qarm-rl
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv qarmrl
   source qarmrl/bin/activate  # Sur macOS/Linux
   # ou
   qarmrl\Scripts\activate     # Sur Windows
   ```
   ou avec conda
   ```bash
   conda create -n qarmrl python
   conda activate qarmrl
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Toutes les configurations se trouvent dans [`core/config.py`](core/config.py).

## Utilisation

### Démarrage du Contrôleur

1. **Ouvrir et lancer le modèle Simulink**
   - Ouvrir `BasicIO_pwm_mode_retest.slx` dans MATLAB
   - Lancer la simulation

2. **Exécuter le script Python**
   ```bash
   python main.py
   ```

3. **Arrêter le contrôleur**
   - Appuyer sur `Ctrl+C` pour un arrêt propre


## Convention des Axes

Point de vue du robot :
- **Base** : `+` = rotation vers la gauche
- **Épaule** : `+` = mouvement vers le bas/avant
- **Coude** : `+` = mouvement vers l'avant
- **Poignet** : `+` = rotation sens horaire
- **Pince** : `1` = fermée, `0` = ouverte

## Debugging

Pour activer les logs de debug détaillés, modifiez dans [`core/logger.py`](core/logger.py) :
```python
logging.basicConfig(
    level=logging.DEBUG,  # Changez INFO en DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```
