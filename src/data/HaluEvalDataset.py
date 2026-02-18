from datasets import load_dataset
from torch.utils.data import Dataset
from src.model.utils import get_weight_dir

REPO_NAME = "pminervini/HaluEval"

class HaluEvalDataset(Dataset):
    def __init__(self, label=0, recreate_ids=True, use_local=False):
        if not use_local:
            dataset_dict = load_dataset(REPO_NAME, "qa")
            # HaluEval might not have standard splits, use the first available split
            if isinstance(dataset_dict, dict):
                # If it's a DatasetDict, get the first available split
                available_splits = list(dataset_dict.keys())
                if available_splits:
                    self.dataset = dataset_dict[available_splits[0]]
                else:
                    raise ValueError(f"No splits found in dataset {REPO_NAME}")
            else:
                # If it's already a Dataset, use it directly
                self.dataset = dataset_dict
        else:
            local_model_path = get_weight_dir(REPO_NAME, repo_type="datasets", subset="qa")
            dataset_dict = load_dataset("parquet", data_dir=local_model_path)
            if isinstance(dataset_dict, dict):
                available_splits = list(dataset_dict.keys())
                if available_splits:
                    self.dataset = dataset_dict[available_splits[0]]
                else:
                    raise ValueError(f"No splits found in local dataset at {local_model_path}")
            else:
                self.dataset = dataset_dict

        if ('instance_id' not in self.dataset.column_names) or recreate_ids:
            self.dataset = self.create_instance_ids()
        
        self.label = label
        self.dataset = self.dataset.shuffle(seed=42)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        id = self.dataset[idx]['instance_id']

        # Check which format we have (dialogue vs QA)
        if 'dialogue_history' in self.dataset.column_names:
            # Original format with dialogue
            history = self.dataset[idx]['dialogue_history']
            knowledge = self.dataset[idx]['knowledge']
            question = history + "\n[Further Knowledge]\n" + knowledge
            
            if self.label == 0:
                answer = self.dataset[idx]['right_response']
            else:
                answer = self.dataset[idx]['hallucinated_response']
        else:
            # QA format (pminervini/HaluEval)
            question_text = self.dataset[idx].get('question', '')
            knowledge = self.dataset[idx].get('knowledge', '')
            
            # Combine question and knowledge
            if knowledge:
                question = question_text + "\n[Further Knowledge]\n" + knowledge
            else:
                question = question_text
            
            if self.label == 0:
                answer = self.dataset[idx]['right_answer']
            else:
                answer = self.dataset[idx]['hallucinated_answer']

        return question, answer, id
    

    def get_language_by_instance_id(self, instance_id):
        return "EN"  # HaluEval is in English, so we return "EN" directly

    
    def create_instance_ids(self):
        instance_ids = list(range(len(self.dataset)))

        if "instance_id" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("instance_id")

        self.dataset = self.dataset.add_column("instance_id", instance_ids)

        return self.dataset
