import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leggi il CSV
df = pd.read_csv('truthXEperiments.csv')

# Configurazione dei colori come nell'immagine
colors = {
    'baseline': '#FF6B6B',           # Rosso
    'truthx_standard': '#4169E1',     # Blu
    'cross_dataset': '#95E1D3',       # Verde chiaro
    'cross_dataset_2': '#B19CD9',     # Viola
    'cross_model1': '#FFB347',        # Arancione (legacy / cross_model1)
    'cross_model2': '#FFB6C1'         # Rosa per i nuovi cross_model2
}

# Dataset labels
dataset_labels = {'belief_bank_facts': 'BBF', 
                  'belief_bank_constraints': 'BBC', 
                  'halu_eval': 'HE'}

def prepare_data_for_model(df, model_name, model_label):
    """Prepara i dati per un specifico modello"""
    model_df = df[df['model'] == model_name].copy()
    
    # Organizza i dati per dataset
    datasets = ['belief_bank_facts', 'belief_bank_constraints', 'halu_eval']
    plot_data = {ds: {} for ds in datasets}
    
    for dataset in datasets:
        dataset_rows = model_df[model_df['dataset'] == dataset]
        
        # Baseline
        baseline = dataset_rows[dataset_rows['type'] == 'Baseline']
        if not baseline.empty:
            plot_data[dataset]['baseline'] = {
                'rate': float(baseline.iloc[0]['hallucination_rate']),
                'id': str(baseline.iloc[0]['experiment_id'])
            }
        
        # TruthX Standard (include anche il typo "Stanadard")
        truthx = dataset_rows[
            (dataset_rows['type'].str.contains('TruthX', na=False, case=False)) | 
            (dataset_rows['type'].str.contains('Stanadard', na=False, case=False))
        ]
        if not truthx.empty:
            plot_data[dataset]['truthx_standard'] = {
                'rate': float(truthx.iloc[0]['hallucination_rate']),
                'id': str(truthx.iloc[0]['experiment_id'])
            }
        
        # Cross-Dataset - prendiamo i primi due se disponibili
        cross_dataset = dataset_rows[dataset_rows['type'] == 'Cross-Dataset']
        if len(cross_dataset) >= 1:
            plot_data[dataset]['cross_dataset'] = {
                'rate': float(cross_dataset.iloc[0]['hallucination_rate']),
                'id': str(cross_dataset.iloc[0]['experiment_id'])
            }
        if len(cross_dataset) >= 2:
            plot_data[dataset]['cross_dataset_2'] = {
                'rate': float(cross_dataset.iloc[1]['hallucination_rate']),
                'id': str(cross_dataset.iloc[1]['experiment_id'])
            }
        
        # CrossModel variants (support 'CrossModel', 'cross_model1', 'cross_model2')
        cm_rows_lower = dataset_rows['type'].astype(str).str.lower()
        # cross_model1 (include legacy 'CrossModel' / 'cross_model')
        cm1 = dataset_rows[(cm_rows_lower == 'cross_model1') | (cm_rows_lower == 'crossmodel') | (cm_rows_lower == 'cross_model')]
        if not cm1.empty:
            plot_data[dataset]['cross_model1'] = {
                'rate': float(cm1.iloc[0]['hallucination_rate']),
                'id': str(cm1.iloc[0]['experiment_id'])
            }
        # cross_model2 (esplicito)
        cm2 = dataset_rows[cm_rows_lower == 'cross_model2']
        if not cm2.empty:
            plot_data[dataset]['cross_model2'] = {
                'rate': float(cm2.iloc[0]['hallucination_rate']),
                'id': str(cm2.iloc[0]['experiment_id'])
            }
    
    return plot_data

def plot_model_results(plot_data, model_label, colors, dataset_labels):
    """Crea il grafico per un modello"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Posizioni per i dataset sull'asse X
    x_pos = np.arange(len(dataset_labels))
    width = 0.15  # Larghezza di ogni barra
    
    # Categorie di esperimenti
    experiment_types = ['baseline', 'truthx_standard', 'cross_dataset', 'cross_dataset_2', 'cross_model1', 'cross_model2']
    # Mostra entrambe le varianti di cross-dataset e cross-model nella legenda
    labels_legend = ['baseline', 'truthx_standard', 'cross_dataset1', 'cross_dataset2', 'cross_model1', 'cross_model2']
    
    # Calcola il valore massimo (per offset delle etichette)
    max_rate = 0.0
    for ds in plot_data:
        for exp in plot_data[ds]:
            r = plot_data[ds][exp]['rate']
            if r > max_rate:
                max_rate = r
    if max_rate == 0:
        max_rate = 1.0
    
    # Plot per ogni tipo di esperimento (salva anche gli experiment_id per le etichette)
    for i, exp_type in enumerate(experiment_types):
        values = []
        ids = []
        positions_list = []
        for j, dataset in enumerate(dataset_labels.keys()):
            if exp_type in plot_data[dataset] and plot_data[dataset][exp_type]['rate'] > 0:
                values.append(plot_data[dataset][exp_type]['rate'])
                ids.append(plot_data[dataset][exp_type]['id'])
                positions_list.append(x_pos[j] + (i - 2) * width)
        
        # Plotta solo se ci sono valori
        if values:
            bars = ax.bar(positions_list, values, width, label=labels_legend[i], 
                         color=colors[exp_type], edgecolor='black', linewidth=0.5)
            # Aggiungi valore sopra ogni barra e experiment_id sotto la barra (rotato 90°)
            for bar, rate, exp_id in zip(bars, values, ids):
                bx = bar.get_x() + bar.get_width() / 2
                bh = bar.get_height()
                # Valore sopra la barra (formattato come percentuale)
                ax.text(bx, bh + max_rate * 0.02, f"{rate*100:.2f}%", ha='center', va='bottom', fontsize=8)
                # experiment_id sotto la barra (fuori dall'area dei dati)
                ax.text(bx, -0.05, exp_id, transform=ax.get_xaxis_transform(), rotation=90, 
                        va='top', ha='center', fontsize=8)
    
    # Configurazione assi e labels
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('Hallucination Rate', fontsize=12)
    ax.set_title(model_label, fontsize=20, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([dataset_labels[ds] for ds in dataset_labels.keys()], 
                        fontsize=16, fontweight='bold')
    
    # Griglia
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Legenda
    handles, labels = ax.get_legend_handles_labels()
    # Rimuovi duplicati dalla legenda
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right', fontsize=10, framealpha=0.9,
             title='legenda', title_fontsize=10)
    
    # Limiti assi (lasciamo lo 0 come minimo)
    ax.set_ylim(0, max_rate * 1.20)
    
    plt.tight_layout()
    return fig

# Prepara dati per Gemma
gemma_data = prepare_data_for_model(df, 'google/gemma-2-9b-it', 'Gemma')

# Prepara dati per Llama
llama_data = prepare_data_for_model(df, 'meta-llama/Llama-3.1-8B-Instruct', 'Llama')

# Crea grafici
fig_gemma = plot_model_results(gemma_data, 'Gemma', colors, dataset_labels)
plt.savefig('gemma_results.png', dpi=300, bbox_inches='tight')
print("Grafico Gemma salvato come 'gemma_results.png'")

fig_llama = plot_model_results(llama_data, 'Llama', colors, dataset_labels)
plt.savefig('llama_results.png', dpi=300, bbox_inches='tight')
print("Grafico Llama salvato come 'llama_results.png'")

plt.show()
