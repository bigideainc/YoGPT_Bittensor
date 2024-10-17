import time
import os
import bittensor as bt
import asyncio
import torch
import uuid
import nest_asyncio
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, setup_chat_format
from typing import Tuple
from template.base.miner import BaseMinerNeuron
from template.protocol import TrainingProtocol
from huggingface_hub import HfApi, login
from utils.HFManager import commit_to_central_repo

# Set up nest_asyncio to allow multiple async loops
nest_asyncio.apply()

class OpenELMTrainingMiner(BaseMinerNeuron):
    def __init__(self, base_model: str = 'apple/OpenELM-270M', 
                 dataset_id: str = 'g-ronimo/oasst2_top4k_en', 
                 epochs: int = 1, batch_size: int = 2, 
                 learning_rate: float = 5e-5, 
                 device: str = 'cuda', 
                 hf_token: str = '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', 
                 job_id: str = str(uuid.uuid4()), 
                 central_repo: str = 'Tobius/yogpt_test'):
        super().__init__()
        self.base_model = base_model
        self.dataset_id = dataset_id
        self.epochs = epochs
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.device = device
        self.hf_token = hf_token
        self.job_id = job_id
        self.central_repo = central_repo
        login(self.hf_token)
        self.initialize_model_and_tokenizer()
        self.initialize_dataset()
        self.setup_trainer()
        self.hf_api = HfApi(token=self.hf_token)

    def initialize_model_and_tokenizer(self):
        # Load model from apple/OpenELM-270M
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            token=self.hf_token,
            use_cache=False
        )

        # Load tokenizer from TinyPixel/Llama-2-7B-bf16-sharded
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyPixel/Llama-2-7B-bf16-sharded",
            trust_remote_code=True,
            use_fast=False,
            token=self.hf_token
        )

        set_seed(42)
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        if self.tokenizer.pad_token in [None, self.tokenizer.eos_token]:
            self.tokenizer.pad_token = self.tokenizer.unk_token

    def initialize_dataset(self):
        # Load dataset g-ronimo/oasst2_top4k_en
        self.dataset = load_dataset(self.dataset_id, use_auth_token=self.hf_token)
        self.train_dataset, self.eval_dataset = self.dataset['train'], self.dataset['test']

    def setup_trainer(self):
        training_arguments = TrainingArguments(
            output_dir=f"out_{self.job_id}",
            run_name=f"openelm_{self.job_id}",
            evaluation_strategy="steps",
            label_names=["labels"],
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=8,
            save_steps=250,
            eval_steps=250,
            logging_steps=10,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            lr_scheduler_type="constant",
            optim='paged_adamw_8bit',
            bf16=False,
            report_to="none",  # No WANDB
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            group_by_length=True,
        )

        self.data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template="user",
            response_template="assistant",
            tokenizer=self.tokenizer,
            mlm=False
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            max_seq_length=2048,
            args=training_arguments,
        )

    async def forward(self, synapse: TrainingProtocol) -> TrainingProtocol:
        try:
            if synapse.training_params:
                self.epochs = synapse.training_params.get('epochs', self.epochs)
                self.batch_size = int(synapse.training_params.get('batch_size', self.batch_size))
                self.learning_rate = synapse.training_params.get('learning_rate', self.learning_rate)
                self.setup_trainer()

            # Start the training process
            train_start_time = time.time()
            train_result = self.trainer.train()
            train_end_time = time.time()

            # Evaluate the model
            eval_result = self.trainer.evaluate()
            final_loss = eval_result['eval_loss']
            train_loss = train_result.training_loss

            # Upload to Hugging Face
            repo_name = f"openelm-{int(time.time())}"
            repo_url = self.hf_api.create_repo(repo_name, private=True)
            self.model.push_to_hub(repo_name, use_auth_token=self.hf_token)

            # Collect metrics
            metrics = {
                'total_epochs': self.epochs,
                'train_loss': train_loss,
                'final_loss': final_loss,
                'training_time': train_end_time - train_start_time,
                'model_repo': repo_url,
                'job_id':self.job_id
            }

            miner_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            central_commit_url = commit_to_central_repo(
                self.hf_token,
                self.central_repo,
                repo_url,
                metrics,
                miner_uid
            )

            # Update synapse
            synapse.loss = final_loss
            synapse.model_hash = repo_url
            # synapse.training_metrics = metrics
            # synapse.training_metrics['central_commit_url'] = central_commit_url

        except Exception as e:
            bt.logging.error(f"Error during training: {str(e)}")
            synapse.loss = None
            synapse.model_hash = None
            # synapse.training_metrics = {} 
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
            self.model = AutoModelForCausalLM.from_pretrained("./model_checkpoint", device_map='auto')

    async def blacklist(self, synapse: TrainingProtocol) -> Tuple[bool, str]:
        return (synapse.dendrite.hotkey not in self.metagraph.hotkeys, 
                "Unrecognized hotkey" if synapse.dendrite.hotkey not in self.metagraph.hotkeys else "Hotkey recognized!")

    async def priority(self, synapse: TrainingProtocol) -> float:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

if __name__ == "__main__":
    miner = OpenELMTrainingMiner(
        base_model='apple/OpenELM-270M',
        dataset_id='g-ronimo/oasst2_top4k_en',  
        epochs=1,
        batch_size=2,
        learning_rate=5e-5,
        device='cuda',
        hf_token="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",
        job_id=str(uuid.uuid4()),
        central_repo="Tobius/yogpt_test",
    )
    
    async def main():
        try:
            await miner.run_training_loop()
        except KeyboardInterrupt:
            bt.logging.info("Miner stopped.")

    asyncio.run(main())
