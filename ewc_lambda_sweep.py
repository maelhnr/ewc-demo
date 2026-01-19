import numpy as np
import matplotlib.pyplot as plt
import torch

# On importe toutes les fonctions du fichier principal
from ewc_mnist import (
    MLP,
    get_dataloaders_3tasks,
    train_one_epoch,
    evaluate,
    EWC_Multi,
    eval_all,
    device
)

def run_lambda_sweep(
    lambdas=[0, 10, 100, 300, 1000, 3000],
    epochs_per_task=3,
    lr=1e-3,
    batch_size=128
):
    """
    Étudie l’impact du paramètre lambda sur l’oubli catastrophique.
    Pour chaque valeur de lambda, le modèle apprend séquentiellement A -> B -> C,
    puis on mesure performance finale sur A, B, C.
    """

    loaders = get_dataloaders_3tasks(batch_size=batch_size)

    acc_A = []
    acc_B = []
    acc_C = []

    for lam in lambdas:
        print(f"\n===== LAMBDA = {lam} =====")

        # reset modèle pour chaque lambda
        model = MLP(hidden_dim=100).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        ewc = EWC_Multi()

        # séquence d’apprentissage A -> B -> C
        for task_name in ["A", "B", "C"]:
            train_loader, _ = loaders[task_name]

            for ep in range(1, epochs_per_task + 1):
                train_one_epoch(
                    model, train_loader, opt,
                    ewc=ewc if len(ewc.tasks) > 0 else None,
                    lambda_ewc=lam
                )

            # on ajoute la tâche au EWC
            ewc.add_task(model, train_loader, fisher_samples=1000)

        # évaluation finale après avoir appris les 3 tâches
        results = eval_all(model, loaders)
        acc_A.append(results["A"] * 100)
        acc_B.append(results["B"] * 100)
        acc_C.append(results["C"] * 100)

        print(f"After tasks A->B->C: A={results['A']*100:.2f}%, "
              f"B={results['B']*100:.2f}%, C={results['C']*100:.2f}%")

    return lambdas, acc_A, acc_B, acc_C


def plot_lambda_sweep(lambdas, acc_A, acc_B, acc_C):
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, acc_A, marker="o", label="Accuracy Task A")
    plt.plot(lambdas, acc_B, marker="o", label="Accuracy Task B")
    plt.plot(lambdas, acc_C, marker="o", label="Accuracy Task C")

    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("Accuracy after learning A→B→C (%)")
    plt.title("Impact du paramètre λ sur la stabilité/plasticité (EWC)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lambda_sweep.png")
    plt.show()


if __name__ == "__main__":
    lambdas = [0, 10, 100, 300, 1000, 3000]
    lambdas, acc_A, acc_B, acc_C = run_lambda_sweep(
        lambdas=lambdas,
        epochs_per_task=3,
        lr=1e-3,
        batch_size=128
    )
    plot_lambda_sweep(lambdas, acc_A, acc_B, acc_C)
