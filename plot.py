import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_json(file_path):
    """Charge et renvoie le contenu d'un fichier JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Dictionnaire des chemins de base pour chaque expérience
    base_paths = {
        'vanila_model': 'experiments/run_basic_loss/cifar10',
        'our_model': 'experiments/run_003/cifar10'
    }
    
    # Conditions : résultats de test et de validation
    conditions = ['test', 'valid']
    
    # Dictionnaire pour stocker les résultats
    results = {exp: {} for exp in base_paths.keys()}
    
    # Charger les résultats depuis les fichiers JSON
    for exp, path in base_paths.items():
        for cond in conditions:
            file_path = os.path.join(path, f"{cond}_result.json")
            results[exp][cond] = load_json(file_path)
    
    # =========================
    # Plot des métriques de base
    # =========================
    metrics = ['nll', 'nll_miss', 'ece', 'top-1', 'top-2', 'top-3']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cond in zip(axes, conditions):
        # Extraire les valeurs pour chaque expérience
        values_run_basic = [results['vanila_model'][cond][m] for m in metrics]
        values_run_001 = [results['our_model'][cond][m] for m in metrics]
        
        index = np.arange(len(metrics))
        bar_width = 0.35
        
        ax.bar(index - bar_width/2, values_run_basic, bar_width, label='vanila_model')
        ax.bar(index + bar_width/2, values_run_001, bar_width, label='our_model')
        
        ax.set_title(f"Base metrics ({cond.capitalize()})")
        ax.set_xticks(index)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('plots/basic_metrics.png')
    plt.show()
    
    # ======================================
    # Plot des métriques d'entropie prédictive
    # ======================================
    entropy_types = ['total', 'aleatoric', 'epistemic']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cond in zip(axes, conditions):
        # Pour vanila_model
        means_basic = [results['vanila_model'][cond]['predictive_entropy'][etype][0] for etype in entropy_types]
        stds_basic = [results['vanila_model'][cond]['predictive_entropy'][etype][1] for etype in entropy_types]
        # Pour our_model
        means_001 = [results['our_model'][cond]['predictive_entropy'][etype][0] for etype in entropy_types]
        stds_001 = [results['our_model'][cond]['predictive_entropy'][etype][1] for etype in entropy_types]
        
        index = np.arange(len(entropy_types))
        bar_width = 0.35
        
        ax.bar(index - bar_width/2, means_basic, bar_width, yerr=stds_basic, capsize=5, label='vanila_model')
        ax.bar(index + bar_width/2, means_001, bar_width, yerr=stds_001, capsize=5, label='our_model')
        
        ax.set_title(f"Predictive entropy ({cond.capitalize()})")
        ax.set_xticks(index)
        ax.set_xticklabels(entropy_types)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('plots/entropy_metrics.png')
    plt.show()

if __name__ == '__main__':
    main()
