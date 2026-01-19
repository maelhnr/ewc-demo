import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ---------- Utils : seed + device ----------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
set_seed(42)


# ---------- Modèle : MLP simple pour MNIST ----------

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------- Dataset : MNIST + Permuted MNIST ----------

class PermutedMNIST(Dataset):
    """
    Wrap d'un dataset MNIST en appliquant une permutation fixe des pixels.
    """
    def __init__(self, base_dataset, permutation):
        self.base_dataset = base_dataset
        self.permutation = permutation  # Tensor de shape (28*28,)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]  # x: (1, 28, 28)
        x = x.view(-1)[self.permutation].view(1, 28, 28)
        return x, y
    

def get_dataloaders(batch_size=128):
    transform = transforms.ToTensor()

    # Task A: MNIST normal
    train_a = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_a = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Task B: MNIST permuté (permutation fixe)
    perm = torch.randperm(28 * 28)
    train_b = PermutedMNIST(train_a, perm)
    test_b = PermutedMNIST(test_a, perm)

    train_loader_a = DataLoader(train_a, batch_size=batch_size, shuffle=True)
    test_loader_a = DataLoader(test_a, batch_size=batch_size, shuffle=False)

    train_loader_b = DataLoader(train_b, batch_size=batch_size, shuffle=True)
    test_loader_b = DataLoader(test_b, batch_size=batch_size, shuffle=False)

    return train_loader_a, test_loader_a, train_loader_b, test_loader_b


def get_dataloaders_3tasks(batch_size=128):
    transform = transforms.ToTensor()

    # Base MNIST
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Task A: MNIST normal
    train_a, test_a = train_base, test_base

    # Task B: permutation #1
    perm_b = torch.randperm(28 * 28)
    train_b = PermutedMNIST(train_base, perm_b)
    test_b  = PermutedMNIST(test_base, perm_b)

    # Task C: permutation #2 (différente)
    perm_c = torch.randperm(28 * 28)
    train_c = PermutedMNIST(train_base, perm_c)
    test_c  = PermutedMNIST(test_base, perm_c)

    loaders = {
        "A": (DataLoader(train_a, batch_size=batch_size, shuffle=True),
              DataLoader(test_a,  batch_size=batch_size, shuffle=False)),
        "B": (DataLoader(train_b, batch_size=batch_size, shuffle=True),
              DataLoader(test_b,  batch_size=batch_size, shuffle=False)),
        "C": (DataLoader(train_c, batch_size=batch_size, shuffle=True),
              DataLoader(test_c,  batch_size=batch_size, shuffle=False)),
    }
    return loaders




# ---------- Fonctions train / test ----------

def train_one_epoch(model, dataloader, optimizer, ewc=None, lambda_ewc=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        ce_loss = F.cross_entropy(outputs, y)

        if ewc is not None and lambda_ewc > 0.0:
            penalty = ewc.penalty(model)
            loss = ce_loss + (lambda_ewc / 2.0) * penalty
        else:
            loss = ce_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = outputs.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


# ---------- Classe EWC : calcule Fisher + pénalité ----------

class EWC:
    def __init__(self, model: nn.Module, dataloader, fisher_samples=2000):
        """
        model: modèle déjà entraîné sur la tâche A
        dataloader: dataloader de la tâche A (train)
        fisher_samples: nombre max d'exemples utilisés pour approximer la Fisher
        """
        self.model = model
        self.fisher = {}
        self.params_star = {}

        # On stocke les paramètres optimaux après la tâche A
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params_star[n] = p.detach().clone()

        # On calcule la diagonale de la Fisher
        self._compute_fisher(dataloader, fisher_samples)

    def _compute_fisher(self, dataloader, fisher_samples):
        self.model.eval()
        # Initialiser Fisher à zéro
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = torch.zeros_like(p, device=device)

        total_samples = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            outputs = self.model(x)
            # Cross-entropy = - log likelihood => les gradients au carré conviennent
            loss = F.cross_entropy(outputs, y)
            loss.backward()

            batch_size = x.size(0)
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += (p.grad.detach() ** 2) * batch_size

            total_samples += batch_size
            if total_samples >= fisher_samples:
                break

        # Moyenne sur le nombre total d'exemples vus
        for n in self.fisher:
            self.fisher[n] /= float(total_samples)

    def penalty(self, model: nn.Module):
        """
        Terme de pénalité EWC = somme_i F_i * (theta_i - theta_i^*)^2
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = self.fisher[n] * (p - self.params_star[n]) ** 2
                loss += _loss.sum()
        return loss

class EWC_Multi:
    def __init__(self):
        # liste de (params_star, fisher) pour chaque tâche déjà apprise
        self.tasks = []

    @torch.no_grad()
    def _snapshot_params(self, model):
        snap = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                snap[n] = p.detach().clone()
        return snap

    def add_task(self, model, dataloader, fisher_samples=2000):
        """
        À appeler à la FIN de chaque tâche, pour stocker θ* et F diag.
        """
        model.eval()
        params_star = self._snapshot_params(model)

        fisher = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p, device=device)

        total = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()

            bs = x.size(0)
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += (p.grad.detach() ** 2) * bs

            total += bs
            if total >= fisher_samples:
                break

        for n in fisher:
            fisher[n] /= float(total)

        self.tasks.append((params_star, fisher))

    def penalty(self, model):
        """
        Somme des pénalités sur toutes les tâches passées.
        """
        loss = 0.0
        for params_star, fisher in self.tasks:
            for n, p in model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - params_star[n]) ** 2).sum()
        return loss

def eval_all(model, loaders):
    results = {}
    for name, (_, test_loader) in loaders.items():
        _, acc = evaluate(model, test_loader)
        results[name] = acc
    return results


# ---------- EXPÉRIENCE COMPLÈTE ----------

def run_experiment(
    epochs_a=5,
    epochs_b=5,
    lr=1e-3,
    lambda_ewc=1000.0,
    batch_size=128,
):
    # 1) Data
    train_a, test_a, train_b, test_b = get_dataloaders(batch_size=batch_size)

    # 2) Tâche A : entraînement de base sur MNIST
    print("=== Tâche A : MNIST (baseline) ===")
    model_a = MLP().to(device)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr)

    for epoch in range(1, epochs_a + 1):
        train_loss, train_acc = train_one_epoch(model_a, train_a, optimizer_a)
        test_loss, test_acc = evaluate(model_a, test_a)
        print(f"[A][Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, "
              f"test_acc={test_acc*100:.2f}%")

    # Sauvegarde des poids après tâche A
    params_after_A = {k: v.detach().clone() for k, v in model_a.state_dict().items()}

    # Évaluation de référence sur A
    base_loss_A, base_acc_A = evaluate(model_a, test_a)
    print(f"\nPerformance de référence sur A (après entraînement A): {base_acc_A*100:.2f}%")

    # 3) Tâche B sans EWC (naïf)
    print("\n=== Tâche B : Permuted MNIST (sans EWC) ===")
    model_naive = MLP().to(device)
    model_naive.load_state_dict(params_after_A)  # on part du modèle appris sur A
    optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=lr)

    for epoch in range(1, epochs_b + 1):
        train_loss, train_acc = train_one_epoch(model_naive, train_b, optimizer_naive)
        test_loss_B, test_acc_B = evaluate(model_naive, test_b)
        print(f"[B naive][Epoch {epoch}] train_loss={train_loss:.4f}, "
              f"test_acc_B={test_acc_B*100:.2f}%")

    # Accuracies finales après avoir appris B sans EWC
    loss_A_naive, acc_A_naive = evaluate(model_naive, test_a)
    loss_B_naive, acc_B_naive = evaluate(model_naive, test_b)
    print(f"\nAprès tâche B SANS EWC :")
    print(f" - Accuracité sur A : {acc_A_naive*100:.2f}%")
    print(f" - Accuracité sur B : {acc_B_naive*100:.2f}%")

    # 4) Tâche B avec EWC
    print("\n=== Tâche B : Permuted MNIST (AVEC EWC) ===")
    model_ewc = MLP().to(device)
    model_ewc.load_state_dict(params_after_A)  # même point de départ que le naïf
    optimizer_ewc = torch.optim.Adam(model_ewc.parameters(), lr=lr)

    # Construire l'objet EWC à partir de la tâche A
    ewc_obj = EWC(model_ewc, train_a, fisher_samples=2000)

    for epoch in range(1, epochs_b + 1):
        train_loss, train_acc = train_one_epoch(
            model_ewc,
            train_b,
            optimizer_ewc,
            ewc=ewc_obj,
            lambda_ewc=lambda_ewc,
        )
        test_loss_B, test_acc_B = evaluate(model_ewc, test_b)
        print(f"[B EWC][Epoch {epoch}] train_loss={train_loss:.4f}, "
              f"test_acc_B={test_acc_B*100:.2f}%")

    loss_A_ewc, acc_A_ewc = evaluate(model_ewc, test_a)
    loss_B_ewc, acc_B_ewc = evaluate(model_ewc, test_b)

    print(f"\nAprès tâche B AVEC EWC :")
    print(f" - Accuracité sur A : {acc_A_ewc*100:.2f}%")
    print(f" - Accuracité sur B : {acc_B_ewc*100:.2f}%")

    # 5) Résumé visuel
    labels = ["Après A", "Après B (naïf)", "Après B (EWC)"]
    acc_A_values = [base_acc_A * 100, acc_A_naive * 100, acc_A_ewc * 100]
    acc_B_values = [0, acc_B_naive * 100, acc_B_ewc * 100]  # 0 = pas entraîné sur B

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, acc_A_values, width, label="Task A")
    plt.bar(x + width/2, acc_B_values, width, label="Task B")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Performance sur les tâches A et B")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ewc_results.png")
    plt.show()

def run_3tasks(epochs_per_task=3, lr=1e-3, lambda_ewc=1000.0, batch_size=128):
    loaders = get_dataloaders_3tasks(batch_size=batch_size)

    tasks_order = ["A", "B", "C"]

    naive_history = []  # liste de dict: [{"A":acc, "B":acc, "C":acc}, ...]
    ewc_history = []

    # ----- 1) Naïf -----
    print("\n===== EXPÉRIENCE NAÏVE (sans EWC) : A -> B -> C =====")
    model_naive = MLP().to(device)
    opt = torch.optim.Adam(model_naive.parameters(), lr=lr)

    for task_name in tasks_order:
        train_loader, _ = loaders[task_name]
        print(f"\n--- Training on Task {task_name} ---")
        for ep in range(1, epochs_per_task + 1):
            tr_loss, tr_acc = train_one_epoch(model_naive, train_loader, opt)
            print(f"[Naive][{task_name}][Ep {ep}] train_acc={tr_acc*100:.2f}%")

        res = eval_all(model_naive, loaders)
        naive_history.append(res)  # on stocke les accuracies
        print(f"Eval after learning {task_name}: " +
              ", ".join([f"{k}:{v*100:.2f}%" for k, v in res.items()]))

    # ----- 2) EWC -----
    print("\n===== EXPÉRIENCE EWC : A -> B -> C =====")
    model_ewc = MLP().to(device)
    opt2 = torch.optim.Adam(model_ewc.parameters(), lr=lr)
    ewc = EWC_Multi()

    for task_name in tasks_order:
        train_loader, _ = loaders[task_name]
        print(f"\n--- Training on Task {task_name} ---")

        for ep in range(1, epochs_per_task + 1):
            tr_loss, tr_acc = train_one_epoch(
                model_ewc, train_loader, opt2,
                ewc=ewc if len(ewc.tasks) > 0 else None,
                lambda_ewc=lambda_ewc
            )
            print(f"[EWC][{task_name}][Ep {ep}] train_acc={tr_acc*100:.2f}%")

        # On ajoute la tâche courante dans l’EWC (snapshot + Fisher)
        ewc.add_task(model_ewc, train_loader, fisher_samples=2000)

        res = eval_all(model_ewc, loaders)
        ewc_history.append(res)
        print(f"Eval after learning {task_name}: " +
              ", ".join([f"{k}:{v*100:.2f}%" for k, v in res.items()]))

    # Après les deux expériences, on trace les courbes
    plot_3tasks_results(tasks_order, naive_history, ewc_history)

def plot_3tasks_results(tasks_order, naive_history, ewc_history):
    """
    tasks_order: ["A","B","C"]
    naive_history: liste de dicts, un par tâche apprise (après A, après B, après C)
    ewc_history: idem pour EWC
    """
    eval_tasks = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for i, eval_task in enumerate(eval_tasks):
        ax = axes[i]
        # accuracies pour cette tâche, après chaque étape d'apprentissage
        naive_acc = [h[eval_task] * 100 for h in naive_history]
        ewc_acc   = [h[eval_task] * 100 for h in ewc_history]

        ax.plot(tasks_order, naive_acc, marker="o", label="Naive")
        ax.plot(tasks_order, ewc_acc, marker="o", label="EWC")

        ax.set_title(f"Tâche évaluée : {eval_task}")
        ax.set_xlabel("Tâche apprise")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True)

    axes[0].legend()
    plt.suptitle("Évolution des performances sur A, B, C (Naive vs EWC)")
    plt.tight_layout()
    plt.savefig("ewc_3tasks_curves.png")
    plt.show()


if __name__ == "__main__":
    run_experiment(
        epochs_a=5,
        epochs_b=5,
        lr=1e-3,
        lambda_ewc=1000.0,
        batch_size=128,
    )

# if __name__ == "__main__":
#     run_3tasks(
#         epochs_per_task=3,
#         lr=1e-3,
#         lambda_ewc=1000.0,
#         batch_size=128
#     )