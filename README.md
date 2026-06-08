# Module de perception / YOLO

Cette branche permet de tester l'intégration de la caméra RGB-D et du modèle de vision YOLO utilisé pour la détection d'objets.

Le modèle détecte les objets présents dans l'image et renvoie, pour chacun d'eux :

* la classe détectée ;
* la bounding box (bbox) ;
* les coordonnées du centre de la bbox ;
* la profondeur estimée grâce à la caméra RGB-D.

## Exécution

1. Lancer le modèle Simulink :

```bash
BasicIO_pwm_mode_retest.slx
```

2. Exécuter le script de test :

```bash
python tests/test_camera_thread.py
```

## Résultat attendu

Une fenêtre s'ouvre et affiche le flux vidéo de la caméra avec les détections YOLO superposées.

Pour chaque objet détecté, le programme affiche :

* les coordonnées image du centre de la bbox ;
* les coordonnées 3D estimées à partir de la profondeur fournie par la caméra RGB-D.

Une heatmap est également générée. Elle associe une forte probabilité au centre de la bounding box détectée, puis décroît progressivement autour de cette zone. Cette représentation pourra être utilisée par la suite comme entrée du RL (voir le papier tossingbot pour référence).
