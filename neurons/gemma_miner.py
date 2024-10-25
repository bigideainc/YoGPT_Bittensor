import time
import os
import bittensor as bt
import asyncio
import nest_asyncio
import torch
import uuid
import shutil
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from typing import Tuple
from template.base.miner import BaseMinerNeuron
from template.protocol import TrainingProtocol
from huggingface_hub import HfApi, login
from utils.HFManager import commit_to_central_repo
from utils.Helper import register_completed_job

nest_asyncio.apply()

class GemmaFineTuningMiner(BaseMinerNeuron):
    def __init__(self, model_name: str = 'google/gemma-2b', dataset_id: str = 'Abirate/english_quotes', epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-4, device: str = 'cuda', hf_token: str = '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', central_repo: str = 'Tobius/yogpt_test',job_id: str = None):
        super().__init__()
        self.base_model = model_name
        self.dataset_id = dataset_id
        self.epochs = epochs
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.device = device
        self.job_id=job_id
        self.hf_token = hf_token
        self.central_repo = central_repo
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.model = self.initialize_model()
        self.data_collator = None  # Placeholder if you need to define a data collator
        self.initialize_dataset()
        self.setup_trainer()
        self.hf_api = HfApi(token=self.hf_token)

    def initialize_model(self):
        # Load LoRA configuration for 4-bit precision
        lora_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load the model and apply LoRA configuration for fine-tuning
        model = AutoModelForCausalLM.from_pretrained(self.base_model, quantization_config=lora_config, trust_remote_code=True).to(self.device)
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16, lora_dropout=0.01)
        return get_peft_model(model, peft_config)

    def initialize_dataset(self):
        try:
            # Load dataset and tokenize
            dataset = load_dataset(self.dataset_id, split="train", token=self.hf_token, trust_remote_code=True)
            self.tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        except Exception as e:
            bt.logging.error(f"Error initializing dataset: {str(e)}")
            raise

    def tokenize_function(self, examples):
        return self.tokenizer(examples["quote"], padding="max_length", truncation=True)

    def setup_trainer(self):
        training_args = TrainingArguments(
            output_dir='./fine-tuned_model',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_steps=1000,
            save_total_limit=2,
            fp16=True,
            optim="paged_adamw_8bit",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer
        )

    async def forward(self, synapse: TrainingProtocol) -> TrainingProtocol:
        try:
            if synapse.training_params:
                self.epochs = synapse.training_params.get('epochs', self.epochs)
                self.batch_size = int(synapse.training_params.get('batch_size', self.batch_size))
                self.learning_rate = synapse.training_params.get('learning_rate', self.learning_rate)
                self.setup_trainer()

            train_start_time = time.time()
            train_result = self.trainer.train()
            train_end_time = time.time()

            final_loss = train_result.training_loss
            repo_name = f"{self.base_model.split('/')[-1]}-finetuned-{self.job_id}-{int(time.time())}"
            repo_url = self.hf_api.create_repo(repo_name, private=False)
            self.model.push_to_hub(repo_name, use_auth_token=self.hf_token)

            miner_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            metrics = {
                'total_epochs': self.epochs,
                'final_loss': final_loss,
                'training_time': train_end_time - train_start_time,
                'model_repo': repo_url,
                'job_id':self.job_id,
                'datasetid':self.dataset_id,
            }

            central_commit_url = commit_to_central_repo(
                self.hf_token,
                self.central_repo,
                repo_url,
                metrics,
                miner_uid,

            )

            synapse.loss = final_loss
            synapse.model_hash = repo_name
            total_training_time= train_end_time - train_start_time
            # register_completed_job(job_id,repo_url,final_loss,final_loss,total_training_time,miner_uid)
        except Exception as e:
            bt.logging.error(f"Error during training: {str(e)}")
            synapse.loss = None
            synapse.model_hash = None
        return synapse

    async def run_training_loop(self):
        while True:
            try:
                dummy_synapse = TrainingProtocol(
                    model_name=self.base_model,
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
            self.model = AutoModelForCausalLM.from_pretrained("./model_checkpoint").to(self.device)

    async def blacklist(self, synapse: TrainingProtocol) -> Tuple[bool, str]:
        return (synapse.dendrite.hotkey not in self.metagraph.hotkeys, 
                "Unrecognized hotkey" if synapse.dendrite.hotkey not in self.metagraph.hotkeys else "Hotkey recognized!")

    async def priority(self, synapse: TrainingProtocol) -> float:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

# if __name__ == "__main__":
#     miner = GemmaFineTuningMiner(
#         base_model='google/gemma-2b',
#         dataset_id='Abirate/english_quotes', 
#         epochs=1, 
#         batch_size=4,  
#         learning_rate=2e-4,
#         device='cuda',
#         hf_token="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", 
#         central_repo="Tobius/yogpt_test",  
#         job_id=str(uuid.uuid4()),
#     )
    
#     async def main():
#         try:
#             await miner.run_training_loop()
#         except KeyboardInterrupt:
#             bt.logging.info("Miner stopped.")

#     asyncio.run(main())
