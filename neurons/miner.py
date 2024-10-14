import time
import os
import bittensor as bt
import asyncio
import nest_asyncio
from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments)
from datasets import load_dataset
from typing import Dict, Any, Tuple
from template.base.miner import BaseMinerNeuron
from template.protocol import TrainingProtocol
from huggingface_hub import HfApi
nest_asyncio.apply()

class TrainingMiner(BaseMinerNeuron):
    def __init__(self, model_type: str = 'openai-community/gpt2', dataset_id: str = 'iohadrubin/wikitext-103-raw-v1', epochs: int = 3, batch_size: int = 4, learning_rate: float = 1e-4, device: str = 'cuda', hf_token: str = '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', central_repo: str = 'Tobius/yogpt_test'):
        super().__init__()
        # Training parameters
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.dataset_id = dataset_id
        self.hf_token = hf_token
        self.central_repo = central_repo
        # Initialize GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_type, pad_token="")
        self.model = GPT2LMHeadModel.from_pretrained(self.model_type, token=self.hf_token).to(self.device)
        # Data collator for GPT-2
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        # Initialize dataset
        self.initialize_dataset()
        
        # Set up trainer
        self.setup_trainer()
        self.hf_api = HfApi(token=self.hf_token)

    def initialize_dataset(self):
        """Initialize and prepare the dataset for training"""
        bt.logging.info(f"Loading dataset: {self.dataset_id}")
        try:
            # Try to load the dataset without specifying a configuration
            self.dataset = load_dataset(self.dataset_id, split="train", token=self.hf_token, trust_remote_code=True)
            bt.logging.info(f"Dataset loaded. Size: {len(self.dataset)}")
            
            if len(self.dataset) == 0:
                raise ValueError("Dataset is empty")
            
            # Check if 'text' column exists
            if 'text' not in self.dataset.column_names:
                # If 'text' column doesn't exist, try to find a suitable text column
                text_columns = [col for col in self.dataset.column_names if isinstance(self.dataset[0][col], str)]
                if text_columns:
                    text_column = text_columns[0]
                    bt.logging.info(f"Using '{text_column}' as the text column")
                    # Rename the column to 'text'
                    self.dataset = self.dataset.rename_column(text_column, 'text')
                else:
                    raise ValueError("Dataset does not contain a suitable text column")
            
            # Tokenize the dataset
            self.tokenized_dataset = self.dataset.map(
                self.tokenize_function, 
                batched=True, 
                remove_columns=self.dataset.column_names
            )
            
            bt.logging.info(f"Dataset prepared. Size: {len(self.tokenized_dataset)}")
            
            # Validate tokenized dataset
            if len(self.tokenized_dataset) == 0:
                raise ValueError("Tokenized dataset is empty")
            
            # Print sample of tokenized data
            bt.logging.info("Sample of tokenized data:")
            bt.logging.info(self.tokenized_dataset[0])
            
        except Exception as e:
            bt.logging.error(f"Error initializing dataset: {str(e)}")
            raise

    def setup_trainer(self):
        """Set up the trainer with current parameters"""
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            report_to="none",
            fp16=False,
            save_steps=500,
            save_total_limit=3,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

    def tokenize_function(self, examples):
        """Tokenizes the input examples using the class's tokenizer."""
        return self.tokenizer(examples["text"], truncation=True,max_length=512)

    async def forward(self, synapse: TrainingProtocol) -> TrainingProtocol:
        """
        Perform the training steps and update synapse with training results.
        """
        try:
            bt.logging.info("Processing training request.")

            # Update training parameters if provided in synapse
            if synapse.training_params:
                self.epochs = synapse.training_params.get('epochs', self.epochs)
                self.batch_size = synapse.training_params.get('batch_size', self.batch_size)
                self.learning_rate = synapse.training_params.get('learning_rate', self.learning_rate)
                self.setup_trainer()  # Reinitialize trainer with new parameters

            # Execute training and capture results
            bt.logging.info("Starting training iteration...")
            train_start_time = time.time()
            train_result = self.trainer.train()
            train_end_time = time.time()
            
            # Log final loss
            final_loss = train_result.training_loss
            bt.logging.info(f"Training iteration completed. Final loss: {final_loss:.4f}")

            repo_name = f"finetuned-gpt2-{int(time.time())}"
            repo_url = self.hf_api.create_repo(repo_name, public=True)
            self.model.push_to_hub(repo_name, use_auth_token=self.hf_token)

            metrics = {
                'total_epochs': self.epochs,
                'final_loss': final_loss,
                'training_time': train_end_time - train_start_time,
                'model_repo': repo_url
            }

            central_commit_url = commit_to_central_repo(
                self.hf_token,
                self.central_repo,
                repo_url,
                metrics
            )
            # Update the synapse response fields
            synapse.loss = final_loss
            synapse.model_hash = repo_name
            synapse.training_metrics = metrics
            synapse.training_metrics['central_commit_url'] = central_commit_url

            print("Training Results:")
            print(synapse.training_metrics)


        except Exception as e:
            bt.logging.error(f"Error during training: {str(e)}")
            synapse.loss = None
            synapse.model_hash = None

        return synapse

    async def run_training_loop(self):
        """Continuous training loop"""
        while True:
            try:
                bt.logging.info("Initiating training iteration...")
                
                dummy_synapse = TrainingProtocol(
                    model_name=self.model_type,
                    batch_data=[],  # Empty as we're using the pre-loaded dataset
                    training_params={
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate
                    }
                )
                
                # Run training iteration
                result_synapse = await self.forward(dummy_synapse)
                
                if result_synapse.loss is not None:
                    bt.logging.info(f"Training iteration completed. Loss: {result_synapse.loss}")
                    self.save_state()  # Save model state after successful training
                
                # Wait for a short period before next iteration
                await asyncio.sleep(60)  # Adjust as needed
                
            except Exception as e:
                bt.logging.error(f"Error in training loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    def save_state(self):
        """Save model state"""
        self.model.save_pretrained("./model_checkpoint")
        # self.model.save_pretrained("path_to_save", push_to_hub=True, repo_name="my-finetuned-model")
        bt.logging.info("Model state saved")

    def load_state(self):
        """Load model state if checkpoint exists"""
        if os.path.exists("./model_checkpoint"):
            self.model = GPT2LMHeadModel.from_pretrained("./model_checkpoint").to(self.device)
            bt.logging.info("Model state loaded from checkpoint")

    async def blacklist(self, synapse: TrainingProtocol) -> Tuple[bool, str]:
        """Check if the incoming request should be blacklisted"""
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unrecognized hotkey"
        return False, "Hotkey recognized!"

    async def priority(self, synapse: TrainingProtocol) -> float:
        """Assign priority to incoming request based on caller's stake"""
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        return priority

# This is the main function, which runs the miner.
if __name__ == "__main__":
    miner = TrainingMiner(
        model_type='openai-community/gpt2',
        dataset_id='iohadrubin/wikitext-103-raw-v1',
        epochs=1,
        batch_size=32,
        learning_rate=1e-4,
        device='cuda',
        hf_token="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",
        central_repo="Tobius/yogpt_test" 
    )
    
    async def main():
        try:
            # Start the continuous training loop
            await miner.run_training_loop()
        except KeyboardInterrupt:
            bt.logging.info("Miner stopped.")

    # Run the async main function
    asyncio.run(main())