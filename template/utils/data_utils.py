# data_utils.py
from typing import List
from datasets import load_dataset
import os

class DataManager:
    def __init__(self, dataset_name: str = "wikitext", subset: str = "wikitext-2-raw-v1"):
        """
        Initialize the DataManager with specified dataset, using Hugging Face authentication if required.
        
        Args:
            dataset_name (str): Name of the dataset to load
            subset (str): Specific subset of the dataset
        """
        # Get Hugging Face token
        self.hf_token = "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"  # Hardcoded for now; ideally from environment
        if not self.hf_token:
            raise ValueError("Hugging Face token is required!")

        # Load dataset
        self.dataset = load_dataset("carlosejimenez/wikitext__wikitext-2-raw-v1", token=self.hf_token)
        self.current_index = 0
        self.dataset_size = len(self.dataset['train'])
        
    def get_batch(self, batch_size: int) -> List[str]:
        """
        Get a batch of training data.
        
        Args:
            batch_size (int): Size of the batch to return
            
        Returns:
            List[str]: A list of text samples
        """
        texts = []
        for _ in range(batch_size):
            if self.current_index >= self.dataset_size:
                self.current_index = 0  # Reset if we've gone through the dataset
            
            text = self.dataset['train'][self.current_index]['text']
            if text.strip():  # Only add non-empty texts
                texts.append(text)
            else:
                # Log if empty text is found
                print(f"Empty text found at index {self.current_index}")
            
            self.current_index += 1

        # Return empty list if no valid batch found
        if not texts:
            print(f"No valid text found in this batch. Check dataset.")
        
        return texts
    
    def get_dataset_size(self) -> int:
        """
        Get the total size of the training dataset.
        
        Returns:
            int: Number of samples in the training dataset
        """
        return self.dataset_size
