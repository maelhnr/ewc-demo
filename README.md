README - Elastic Weight Consolidation (EWC) Demo

============= Objectif du projet =============

Ce projet a été réalisé dans le cadre de l’évaluation du cours de Deep Learning 2025–2026 (ISAE-Supaero).
L’objectif est de :

- Expliquer clairement le phénomène d’oubli catastrophique dans l’apprentissage séquentiel.
- Implémenter Elastic Weight Consolidation (EWC) sur MNIST et Permuted MNIST.
- Fournir une démonstration de code simple et réutilisable.
- Proposer une extension originale au-delà de la reproduction du papier de Kirkpatrick et al. (2017).

Le projet comprend :

- une expérience principale (2 ou 3 tâches)
- une simulation longue (5 tâches)
- une extension expérimentale (effet du paramètre lambda)

📦 Structure du repository
ewc-demo/
│
├── ewc_mnist.py              # Expériences principales : 2 tâches et 3 tâches
├── ewc_lambda_sweep.py       # Extension originale : étude systématique du paramètre lambda
├── ewc_5tasks_demo.py        # Apprentissage séquentiel long (5 tâches)
│
├── ewc_results.png           # Résultats visuels pour 2 tâches
├── ewc_3tasks_curves.png     # Courbes d’évolution pour 3 tâches
├── ewc_5tasks_heatmap.png    # Heatmap complète pour 5 tâches
├── lambda_sweep.png          # Courbes d’évolution pour 3 tâches en fonction de lambda
│
├── requirements.txt
└── README.md

============= Installation =============

1. Créer un environnement virtuel

python -m venv .venv

3. Activer l’environnement

source venv\Scripts\activate

5. Installer les dépendances

pip install -r requirements.txt

============= 1. Expérience principale : EWC sur deux ou trois tâches =============

Le fichier ewc_mnist.py permet d’exécuter :

- un apprentissage séquentiel classique (naïf) ou avec EWC
- avec deux tâches : MNIST → Permuted MNIST
- ou trois tâches : MNIST → Permuted1 → Permuted2

Il génère automatiquement :

- des résultats chiffrés en console
- des graphiques comparant naive vs EWC

▶️ Lancer l’expérience 2 tâches

Dans ewc_mnist.py, laisser activé :

if __name__ == "__main__":
    run_experiment(
        epochs_a=5,
        epochs_b=5,
        lr=1e-3,
        lambda_ewc=1000.0,
        batch_size=128
    )


Puis exécuter :
python ewc_mnist.py

▶️ Lancer l’expérience 3 tâches

Décommenter :
run_3tasks(...)

Et exécuter :
python ewc_mnist.py

============= 2. Extension originale : étude du paramètre lambda =============

Le fichier ewc_lambda_sweep.py explore plusieurs valeurs de lambda :

lambda ∈ {0, 10, 100, 300, 1000, 3000}

Pour chaque lambda, le modèle apprend successivement : A → B → C

Et on mesure : accuracy sur A, accuracy sur B et accuracy sur C.

Le script génère le graphique lambda_sweep.png, montrant l’impact du lambda sur la stabilité et la plasticité du modèle.

▶️ Lancer l’expérience
python ewc_lambda_sweep.py

============= 3. Simulation avancée : apprentissage long sur 5 tâches =============

Le fichier ewc_5tasks_demo.py montre comment EWC se comporte dans un contexte plus réaliste de continual learning avec :

- T1 : MNIST normal
- T2–T5 : MNIST permuté (4 permutations différentes)

Le script compare :

- modèle naïf (lambda = 0)
- EWC (lambda = 1000)

Il génère une heatmap : ewc_5tasks_heatmap.png

Cette heatmap montre un oubli cumulatif très fort dans le modèle naïf et une stabilité remarquable avec EWC (presque aucune dégradation après 5 tâches).

▶️ Lancer l’expérience
python ewc_5tasks_demo.py

============= Résumé pédagogique =============

1. L’oubli catastrophique

Lorsqu’un réseau apprend des tâches séquentielles, les nouvelles mises à jour de gradient écrasent les connaissances nécessaires aux anciennes tâches.
Résultat : perte massive de performance sur les premières tâches.

2. Le principe d’EWC

EWC ajoute une pénalité quadratique sur les poids les plus importants (estimés avec la matrice de Fisher) :

Loss_total = Loss_task_B + (lambda/2) * Σ[F_i * (θ_i - θ_i*)²]

Cela force le réseau à conserver les paramètres critiques pour les anciennes tâches (stabilité) et apprendre les nouvelles tâches via les paramètres moins critiques (plasticité).

3. Ce que montrent les expériences

Expérience 2 tâches

- modèle naïf : perte sévère sur la tâche A
- modèle EWC : protection très forte de la performance

Expérience 3 tâches

- mise en évidence du compromis stabilité-plasticité
- meilleure compréhension du rôle de lambda

Extension : sweep de lambda

- lambda trop faible → oubli important
- lambda intermédiaire (300–1000) → compromis optimal
- lambda très fort → modèle trop rigide

Expérience 5 tâches

- modèle naïf → effondrement cumulatif (T1 tombe à 35%)
- EWC → toutes les tâches restent autour de 93–96%
- démonstration claire de continual learning stabilisé
