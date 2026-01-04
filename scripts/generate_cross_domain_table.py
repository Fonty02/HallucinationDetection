import json
import os
import glob

base_path = "allucinationDetection/notebooks/nonLinearApproach/approach3OneForAll"
folders = ["LLama_Gemma_BBC", "LLama_Gemma_BBF", "LLama_Gemma_HE"]

print("\\begin{table}[H]")
print("\\centering")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{lllccccccc}")
print("\\toprule")
print("Train Set & Test Set & Teacher & Student & Layer & Acc (T) & AUROC (T) & Acc (S) & AUROC (S) \\\\")
print("\\midrule")

dataset_map = {
    "belief_bank_constraints": "BBC",
    "belief_bank_facts": "BBF",
    "halu_eval": "HE"
}

model_map = {
    "Llama-3.1-8B-Instruct": "Llama",
    "gemma-2-9b-it": "Gemma"
}

for folder in folders:
    path = os.path.join(base_path, folder, "results_metrics")
    files = glob.glob(os.path.join(path, "cross_dataset_eval*.json"))
    
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Data is a list of results for different layers
        for entry in data:
            train_ds = dataset_map.get(entry['dataset_train'], entry['dataset_train'])
            test_ds = dataset_map.get(entry['dataset_head'], entry['dataset_head']) # dataset_head is the test set here
            teacher = model_map.get(entry['teacher_model'], entry['teacher_model'])
            student = model_map.get(entry['student_model'], entry['student_model'])
            layer = entry['layer_type']
            
            metrics_t = entry['eval']['teacher_on_eval']
            metrics_s = entry['eval']['student_adapter_on_eval']
            
            acc_t = metrics_t['accuracy'] * 100
            auroc_t = metrics_t['auroc'] * 100
            acc_s = metrics_s['accuracy'] * 100
            auroc_s = metrics_s['auroc'] * 100
            
            print(f"{train_ds} & {test_ds} & {teacher} & {student} & {layer} & {acc_t:.1f} & {auroc_t:.1f} & {acc_s:.1f} & {auroc_s:.1f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\caption{Risultati Cross-Domain (One-For-All). T=Teacher (Baseline), S=Student (Projected)}")
print("\\label{tab:cross-domain-results}")
print("\\end{table}")
