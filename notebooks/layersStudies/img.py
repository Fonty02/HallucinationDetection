import json
import matplotlib.pyplot as plt
import os

def plot_accuracy_from_json(json_data, model_name=None, dataset="Dataset"):
    """
    Genera un grafico dell'accuracy per layer.
    
    Args:
        json_data (dict): Il dizionario caricato dal file JSON.
        model_name (str): Il nome del modello da plottare.
                          Se None, prende il primo modello trovato nel JSON.
        dataset (str): Nome del dataset per il titolo del file.
    """
    
    # 1. Selezione del modello
    if model_name is None:
        model_name = list(json_data.keys())[0]
    
    if model_name not in json_data:
        print(f"Errore: Modello '{model_name}' non trovato nel JSON.")
        return

    data = json_data[model_name]

    # 2. Configurazione dello Stile
    plt.rcParams.update({
        "font.family": "serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
        "lines.linewidth": 2
    })

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Mappatura colori
    styles = {
        "hidden": {"color": "red", "label": "hidden"},
        "mlp":    {"color": "blue", "label": "mlp"},
        "attn":   {"color": "green", "label": "attn"}
    }

    # 3. Estrazione e Ordinamento dei dati
    for key in ["hidden", "mlp", "attn"]:
        if key in data:
            points = data[key]
            sorted_points = sorted(points, key=lambda x: x['layer'])
            layers = [item['layer'] for item in sorted_points]
            accuracies = [item['accuracy'] for item in sorted_points]
            ax.plot(layers, accuracies, 
                    color=styles[key]["color"], 
                    label=styles[key]["label"])

    # 4. Rifinitura Grafica
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} - {dataset}")
    ax.grid(True, linestyle='-', alpha=1.0)
    
    legend = ax.legend(title="activation", loc="upper left", frameon=True)
    plt.setp(legend.get_title(), fontweight='bold')

    plt.tight_layout()
    
    # Crea la cartella img se non esiste
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/{model_name}_{dataset}_activations.png")
    # plt.show()  # Commentato per esecuzione in ambienti non grafici

content = json.load(open('all_layers_sorted.json'))

for model_name in content.keys():
    plot_accuracy_from_json(content, model_name, "BeliefBank")
