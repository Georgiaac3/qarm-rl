# RL Environments

Ce dossier contient l'environnement Gymnasium basé sur Genesis pour entraîner un agent RL à ajuster les gains PID du QArm.

## Contenu

- `qarm_pid_gym_env.py` : environnement Gymnasium/Genesis.
- `train_sac.py` : script d'entraînement Stable-Baselines3 avec SAC.
- `training_history.py` : utilitaires pour relire les logs TensorBoard et reconstruire une courbe de reward.
- `requirements.txt` : dépendances spécifiques au RL et à Genesis.

## Pré-requis

- Python 3.8 à 3.13.
- Installer les dépendances du projet racine, puis celles du dossier RL.

Depuis la racine du projet :

```bash
pip install -e .
pip install -r rl_envs/requirements.txt
```

## Ce que fait l'environnement

L'environnement `QArmPIDGymEnv` fonctionne ainsi :

- une action RL = 9 valeurs correspondant aux diagonales de `Kp`, `Kd` et `Ki`;
- un `step()` Gymnasium lance une rollout complète dans Genesis;
- chaque rollout exécute une mission carré puis une mission cercle;
- la reward dépend de l'erreur de suivi cartésien, avec un poids supplémentaire sur l'erreur finale.

Le but de l'algorithme RL est donc d'améliorer les gains PID d'une episode à la suivante, pas à chaque pas de physique Genesis.

## Lancer un entraînement SAC

Depuis la racine du projet :

```bash
python rl_envs/train_sac.py --timesteps 20000
```

Options utiles :

```bash
python rl_envs/train_sac.py --timesteps 50000 --log-dir runs/sac_qarm_pid
python rl_envs/train_sac.py --timesteps 20000 --no-randomize-missions
```

Pendant l'entraînement :

- les logs TensorBoard sont écrits dans `runs/sac_qarm_pid/` par défaut;
- les checkpoints intermédiaires sont enregistrés dans `runs/sac_qarm_pid/checkpoints/`.

## Visualiser l'évolution du reward après coup

Le helper `rl_envs/training_history.py` permet de relire les logs TensorBoard après l'entraînement.

Exemple :

```bash
python -c "from rl_envs.training_history import build_reward_plot; build_reward_plot('runs/sac_qarm_pid', show=True)"
```

Pour sauvegarder la figure :

```bash
python -c "from rl_envs.training_history import build_reward_plot; build_reward_plot('runs/sac_qarm_pid', save_path='runs/sac_qarm_pid/reward_curve.png', show=False)"
```

Pour lisser la courbe :

```bash
python -c "from rl_envs.training_history import build_reward_plot; build_reward_plot('runs/sac_qarm_pid', smooth_window=20, show=True)"
```

## Remarques

- `train_sac.py` utilise SAC et un `CheckpointCallback`.
- `QArmPIDGymEnv` peut randomiser le centre et la taille des missions à chaque `reset()`.
