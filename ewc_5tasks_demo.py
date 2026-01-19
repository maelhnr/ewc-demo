import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from ewc_mnist import (
    MLP,
    PermutedMNIST,
    train_one_epoch,
    evaluate,
    EWC_Multi,
    device
)

def make_5tasks_loaders(batch_size=128, n_tasks=5):
    """
    Task 1 = MNIST normal
    Task 2..n = MNIST permuté avec permutations différentes
    """
    transform = transforms.ToTensor()
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    loaders = {}

    # Task 1: normal MNIST
    loaders["T1"] = (
        DataLoader(train_base, batch_size=batch_size, shuffle=True),
        DataLoader(test_base, batch_size=batch_size, shuffle=False)
    )

    # Tasks 2..n: permuted MNIST
    for t in range(2, n_tasks + 1):
        perm = torch.randperm(28 * 28)
        train_t = PermutedMNIST(train_base, perm)
        test_t = PermutedMNIST(test_base, perm)
        loaders[f"T{t}"] = (
            DataLoader(train_t, batch_size=batch_size, shuffle=True),
            DataLoader(test_t, batch_size=batch_size, shuffle=False)
        )

    return loaders


def eval_all_tasks(model, loaders):
    res = {}
    for name, (_, test_loader) in loaders.items():
        _, acc = evaluate(model, test_loader)
        res[name] = acc * 100
    return res


def run_sequence(loaders, lambda_ewc=0, epochs_per_task=3, lr=1e-3, fisher_samples=1000, hidden_dim=100):
    """
    Entraînement séquentiel sur toutes les tâches (T1..Tn)
    Retourne l'historique des accuracies après chaque tâche.
    """
    task_names = list(loaders.keys())
    model = MLP(hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ewc = EWC_Multi()

    history = []  # liste de dicts : accuracies sur toutes les tâches après chaque étape

    for i, task in enumerate(task_names):
        train_loader, _ = loaders[task]

        for ep in range(epochs_per_task):
            train_one_epoch(
                model, train_loader, opt,
                ewc=ewc if (lambda_ewc > 0 and len(ewc.tasks) > 0) else None,
                lambda_ewc=lambda_ewc
            )

        # consolider la tâche qu'on vient d'apprendre (pour EWC)
        if lambda_ewc > 0:
            ewc.add_task(model, train_loader, fisher_samples=fisher_samples)

        # évaluer sur toutes les tâches
        res = eval_all_tasks(model, loaders)
        history.append(res)

        print(f"After learning {task} (lambda={lambda_ewc}): " +
              ", ".join([f"{k}:{v:.2f}%" for k, v in res.items()]))

    return task_names, history


def plot_5tasks(task_names, history_naive, history_ewc, outpath="ewc_5tasks.png"):
    """
    Trace une heatmap: lignes = tâche apprise, colonnes = tâche évaluée.
    """
    n = len(task_names)

    def hist_to_matrix(history):
        M = np.zeros((n, n))
        for i in range(n):  # après apprentissage Ti
            for j, tname in enumerate(task_names):  # éval sur Tj
                M[i, j] = history[i][tname]
        return M

    M_naive = hist_to_matrix(history_naive)
    M_ewc = hist_to_matrix(history_ewc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, M, title in [
        (axes[0], M_naive, "Naïf (λ=0)"),
        (axes[1], M_ewc, "EWC (λ=1000)")
    ]:
        im = ax.imshow(M, vmin=0, vmax=100)
        ax.set_title(title)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(task_names)
        ax.set_yticklabels([f"après {t}" for t in task_names])
        ax.set_xlabel("Tâche évaluée")
        ax.set_ylabel("Étape d'apprentissage")

        # valeurs dans les cases
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    print("Saved:", outpath)


if __name__ == "__main__":
    loaders = make_5tasks_loaders(batch_size=128, n_tasks=5)

    task_names, hist_naive = run_sequence(
        loaders, lambda_ewc=0, epochs_per_task=3, lr=1e-3, fisher_samples=1000, hidden_dim=100
    )

    _, hist_ewc = run_sequence(
        loaders, lambda_ewc=1000, epochs_per_task=3, lr=1e-3, fisher_samples=1000, hidden_dim=100
    )

    plot_5tasks(task_names, hist_naive, hist_ewc, outpath="ewc_5tasks_heatmap.png")
