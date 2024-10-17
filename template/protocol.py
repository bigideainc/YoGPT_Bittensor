import typing
import bittensor as bt
from typing import List, Optional, Dict

class TrainingProtocol(bt.Synapse):
    """Protocol for GPT model training in Bittensor network.
    
    This class encapsulates the necessary parameters and response fields 
    for managing training requests and responses.
    """
    
    # Required request fields
    model_name: str  # Name of the model (e.g., 'gpt2', 'llama2')
    batch_data: List[str]  # Training data batch
    training_params: Dict[str, Optional[float]]  # Hyperparameters for training

    # Response fields
    loss: Optional[float] = None  # Final loss after training
    model_hash: Optional[str] = None  # Hash of the trained model
    training_metrics: Optional[Dict[str, float]] = None  # Metrics related to training

    def deserialize(self) -> 'TrainingProtocol':
        """Deserialize incoming request and return an instance of TrainingProtocol.
        
        This method can be extended to include deserialization logic if necessary.
        """
        return self

    def serialize(self) -> dict:
        """Serialize the response fields for sending back.
        
        Returns:
            dict: A dictionary containing serialized response fields.
        """
        response = {
            "loss": self.loss,
            "model_hash": self.model_hash,
            "training_metrics": self.training_metrics
        }
        return {key: value for key, value in response.items() if value is not None}
