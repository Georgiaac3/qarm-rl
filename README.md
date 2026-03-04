# QARM-RL

Système de contrôle du bras robotique QARM via communication UDP avec Simulink. Implémente une séquence de lancer d'objet avec contrôle proportionnel des vitesses articulaires.

## Structure du Projet

```
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

- Python 3.8 ou supérieur
- MATLAB/Simulink (pour le modèle du robot)

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone <url-du-repo>
   cd qarm-rl
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur macOS/Linux
   # ou
   venv\Scripts\activate     # Sur Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer Pre-commit** (recommandé)
   ```bash
   pre-commit install
   ```

   Les hooks pre-commit s'exécuteront automatiquement avant chaque commit pour :
   - Formater le code avec Black
   - Trier les imports avec isort
   - Vérifier la qualité du code avec Flake8
   - Valider les types avec mypy
   - Nettoyer les fichiers (espaces, fins de ligne, etc.)

   Pour exécuter manuellement sur tous les fichiers :
   ```bash
   pre-commit run --all-files
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

## Développement

### Pre-commit Hooks

Le projet utilise pre-commit pour maintenir la qualité du code. Les hooks configurés :

- **trailing-whitespace** : Supprime les espaces en fin de ligne
- **end-of-file-fixer** : Assure une ligne vide en fin de fichier
- **black** : Formatage automatique du code Python (ligne max: 100 caractères)
- **isort** : Tri automatique des imports
- **flake8** : Vérification de la qualité du code (PEP8)
- **mypy** : Vérification des types statiques

### Commandes utiles

```bash
# Installer les hooks (à faire une fois)
pre-commit install

# Exécuter sur tous les fichiers
pre-commit run --all-files

# Exécuter sur les fichiers modifiés
pre-commit run

# Mettre à jour les hooks vers les dernières versions
pre-commit autoupdate

# Formater un fichier spécifique avec Black
black main.py

# Vérifier les types avec mypy
mypy main.py core/ utils/
```

### Structure de code recommandée

- Utiliser les type hints pour toutes les fonctions
- Documenter avec des docstrings (format Google/NumPy)
- Limiter les lignes à 100 caractères
- Suivre PEP8 pour le style de code
