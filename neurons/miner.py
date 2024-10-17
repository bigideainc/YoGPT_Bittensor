import time
import os
import bittensor as bt
import asyncio
import nest_asyncio
from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments)
from datasets import load_dataset
from typing import Tuple
from template.base.miner import BaseMinerNeuron
from template.protocol import TrainingProtocol
from huggingface_hub import HfApi
from utils.HFManager import commit_to_central_repo

nest_asyncio.apply()

class TrainingMiner(BaseMinerNeuron):
    def __init__(self, model_type: str = 'openai-community/gpt2', dataset_id: str = 'iohadrubin/wikitext-103-raw-v1', epochs: int = 1, batch_size: int = 16, learning_rate: float = 5e-5, device: str = 'cuda', hf_token: str = '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', central_repo: str = 'Tobius/yogpt_test'):
        super().__init__()
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.dataset_id = dataset_id
        self.hf_token = hf_token
        self.central_repo = central_repo
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_type, pad_token="")
        self.model = GPT2LMHeadModel.from_pretrained(self.model_type, token=self.hf_token).to(self.device)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        self.initialize_dataset()
        self.setup_trainer()
        self.hf_api = HfApi(token=self.hf_token)

    def initialize_dataset(self):
        try:
            self.dataset = load_dataset(self.dataset_id, split="train", token=self.hf_token, trust_remote_code=True)
            if 'text' not in self.dataset.column_names:
                text_columns = [col for col in self.dataset.column_names if isinstance(self.dataset[0][col], str)]
                if text_columns:
                    self.dataset = self.dataset.rename_column(text_columns[0], 'text')
                else:
                    raise ValueError("Dataset does not contain a suitable text column")
            
            self.tokenized_dataset = self.dataset.map(
                self.tokenize_function, 
                batched=True, 
                remove_columns=self.dataset.column_names
            )
        except Exception as e:
            bt.logging.error(f"Error initializing dataset: {str(e)}")
            raise

    def setup_trainer(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=500,
            save_steps=1000,
            save_total_limit=2,
            report_to="none",
            fp16=True,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, max_length=512)

    async def forward(self, synapse: TrainingProtocol) -> TrainingProtocol:
        try:
            if synapse.training_params:
                self.epochs = synapse.training_params.get('epochs', self.epochs)
                self.batch_size = synapse.training_params.get('batch_size', self.batch_size)
                self.learning_rate = synapse.training_params.get('learning_rate', self.learning_rate)
                self.setup_trainer()

            train_start_time = time.time()
            train_result = self.trainer.train()
            train_end_time = time.time()
            
            final_loss = train_result.training_loss
            repo_name = f"finetuned-gpt2-{int(time.time())}"
            repo_url = self.hf_api.create_repo(repo_name, private=True)
            self.model.push_to_hub(repo_name, use_auth_token=self.hf_token)

            miner_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
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
                metrics,
                miner_uid
            )

            synapse.loss = final_loss
            synapse.model_hash = repo_name
            synapse.training_metrics = metrics
            synapse.training_metrics['central_commit_url'] = central_commit_url

        except Exception as e:
            bt.logging.error(f"Error during training: {str(e)}")
            synapse.loss = None
            synapse.model_hash = None

        return synapse

    async def run_training_loop(self):
        while True:
            try:
                dummy_synapse = TrainingProtocol(
                    model_name=self.model_type,
                    batch_data=[],
                    training_params={
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate
                    }
                )
                
                result_synapse = await self.forward(dummy_synapse)
                
                if result_synapse.loss is not None:
                    self.save_state()
                
                await asyncio.sleep(300)
                
            except Exception as e:
                bt.logging.error(f"Error in training loop: {str(e)}")
                await asyncio.sleep(300)

    def save_state(self):
        self.model.save_pretrained("./model_checkpoint")

    def load_state(self):
        if os.path.exists("./model_checkpoint"):
            self.model = GPT2LMHeadModel.from_pretrained("./model_checkpoint").to(self.device)

    async def blacklist(self, synapse: TrainingProtocol) -> Tuple[bool, str]:
        return (synapse.dendrite.hotkey not in self.metagraph.hotkeys, 
                "Unrecognized hotkey" if synapse.dendrite.hotkey not in self.metagraph.hotkeys else "Hotkey recognized!")

    async def priority(self, synapse: TrainingProtocol) -> float:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

if __name__ == "__main__":
    miner = TrainingMiner(
        model_type='openai-community/gpt2',
        dataset_id='iohadrubin/wikitext-103-raw-v1',
        epochs=1,
        batch_size=16,
        learning_rate=5e-5,
        device='cuda',
        hf_token="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",
        central_repo="Tobius/yogpt_test" 
    )
    
    async def main():
        try:
            await miner.run_training_loop()
        except KeyboardInterrupt:
            bt.logging.info("Miner stopped.")

    asyncio.run(main())