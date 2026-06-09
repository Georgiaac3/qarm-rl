# QARM-RL

Contrôle du bras robotique QARM avec une interface temps réel et plusieurs routines de démonstration.

## Vue d’ensemble

Le projet est organisé autour de deux parties :

- le **lancement** (`main.py`), qui démarre une routine et l’interface de visualisation ;
- le **package source** (`src/robot_control/`), qui regroupe le contrôle, les missions, les robots, l’UI et les utilitaires.

## Structure du projet

```text
qarm-rl/
├── main.py
├── BasicIO_pwm_mode_retest.slx
├── config.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── routines/
│   ├── base_routine.py
│   ├── engine.py
│   ├── routine1.py
│   └── routine2.py
├── rl_envs/
│   └── qarm_pid_gym_env.py
├── src/
│   └── robot_control/
│       ├── core/
│       ├── missions/
│       ├── robots/
│       ├── ui/
│       └── utils/
└── tests/
    ├── test_types.py
    └── custom_tests/
```

## Rôle des dossiers

- `main.py` : point d’entrée CLI, lance une routine avec `--routine`.
- `routines/` : séquences d’exécution de haut niveau pour le robot.
- `src/robot_control/core/` : logique de base du contrôle, cinématique et dynamique.
- `src/robot_control/missions/` : missions et trajectoires.
- `src/robot_control/robots/` : contrôleurs QARM réel/simulé.
- `src/robot_control/ui/` : dashboard Dear PyGui.
- `src/robot_control/utils/` : types, calculs, trajectoires et logs.
- `rl_envs/` : environnements de simulation / RL.

## Installation

### Prérequis

- Python 3.8+
- MATLAB/Simulink si vous utilisez le modèle `.slx`

### Installation du projet

```bash
pip install -e .
```

Si besoin, installez aussi les dépendances listées dans `requirements.txt`.

## Utilisation

### Lancer une routine

La routine par défaut est la 1.

```bash
python main.py
```

Pour choisir explicitement la routine :

```bash
python main.py --routine 1
python main.py --routine 2
```

### Avec Simulink

1. Ouvrir `BasicIO_pwm_mode_retest.slx` dans MATLAB.
2. Lancer la simulation.
3. Lancer ensuite le script Python.

## Configuration

La configuration globale est centralisée dans `config.py` à la racine et dans les modules de `src/robot_control/`.

## Développement

Commandes utiles :

```bash
pre-commit install
pre-commit run --all-files
black main.py src/ routines/
pytest
```

## Notes

- L’interface graphique est dans `src/robot_control/ui/dashboard.py`.
- Les routines disponibles sont actuellement `routine1` et `routine2`.
