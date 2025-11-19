from datasets import load_dataset
from torch.utils.data import Dataset
from src.model.utils import get_weight_dir

REPO_NAME = "google/simpleqa-verified"

class SimpleQADataset(Dataset):
    def __init__(self, recreate_ids=True, use_local=False):
        """
        SimpleQA Dataset Loader for hallucination detection.
        
        Args:
            recreate_ids (bool): Whether to recreate instance IDs
            use_local (bool): Whether to use locally cached dataset
        """
        if not use_local:
            self.dataset = load_dataset(REPO_NAME, "simpleqa_verified")['eval']
        else:
            local_model_path = get_weight_dir(REPO_NAME, repo_type="datasets")
            self.dataset = load_dataset("parquet", data_dir=local_model_path)['eval']

        if ('instance_id' not in self.dataset.column_names) or recreate_ids:
            self.dataset = self.create_instance_ids()

        self.dataset = self.dataset.shuffle(seed=42)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        """
        Returns:
            question (str): The problem/question
            answer (str): The gold answer
            instance_id (int): Unique instance identifier
        """
        id = self.dataset[idx]['instance_id']
        question = self.dataset[idx]['problem']
        answer = self.dataset[idx]['answer']

        return question, answer, id
    

    def get_language_by_instance_id(self, instance_id):
        """SimpleQA is in English only"""
        return "EN"

    
    def create_instance_ids(self):
        """Create sequential instance IDs for the dataset"""
        instance_ids = list(range(len(self.dataset)))

        if "instance_id" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("instance_id")

        self.dataset = self.dataset.add_column("instance_id", instance_ids)

        return self.dataset


    def save_dataset_as_jsonl(self, output_path):
        """Save the dataset as JSONL format"""
        self.dataset.to_json(output_path, lines=True, orient="records")


    def get_metadata(self, instance_id):
        """
        Get additional metadata for an instance.
        
        Returns:
            dict: Contains topic, answer_type, multi_step, requires_reasoning, urls
        """
        matches = self.dataset.filter(lambda x: x["instance_id"] == instance_id)

        if len(matches) == 0:
            raise ValueError(f"Instance ID {instance_id} not found.")
        
        item = matches[0]
        return {
            "topic": item.get("topic", ""),
            "answer_type": item.get("answer_type", ""),
            "multi_step": item.get("multi_step", False),
            "requires_reasoning": item.get("requires_reasoning", False),
            "urls": item.get("urls", [])
        }
