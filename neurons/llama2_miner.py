import time
import os
import bittensor as bt
import asyncio
import nest_asyncio
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
                          BitsAndBytesConfig, DataCollatorForSeq2Seq)
from datasets import load_dataset
from typing import Tuple
from template.base.miner import BaseMinerNeuron
from template.protocol import TrainingProtocol
from huggingface_hub import HfApi, login
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import torch
from utils.HFManager import commit_to_central_repo

nest_asyncio.apply()

class Llama2TrainingMiner(BaseMinerNeuron):
    def __init__(self, model_name: str = 'NousResearch/Llama-2-7b-chat-hf', 
                 dataset_id: str = 'mlabonne/guanaco-llama2-1k', 
                 epochs: int = 1, batch_size: int = 2, 
                 learning_rate: float = 2e-5, 
                 device: str = 'cuda', 
                 hf_token: str = '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', 
                 central_repo: str = 'Tobius/yogpt_test'):
        super().__init__()
        self.model_name = model_name
        self.dataset_id = dataset_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.hf_token = hf_token
        self.central_repo = central_repo
        login(self.hf_token)
        self.initialize_model_and_tokenizer()
        self.initialize_dataset()
        self.setup_trainer()
        self.hf_api = HfApi(token=self.hf_token)

    def initialize_model_and_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_auth_token=self.hf_token, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            use_auth_token=self.hf_token,
            trust_remote_code=True,
            device_map='auto'
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

    def initialize_dataset(self):
        self.dataset = load_dataset(self.dataset_id, split="train", use_auth_token=self.hf_token)
        self.train_dataset, self.eval_dataset = self.dataset.train_test_split(test_size=0.1).values()

    def setup_trainer(self):
        def find_all_linear_names(model):
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        lora_modules = find_all_linear_names(self.model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=lora_modules
        )

        self.model = get_peft_model(self.model, peft_config)

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            fp16=True,
            logging_steps=10,
            optim="paged_adamw_8bit",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=50,
            output_dir="./results",
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard"  # Changed from "wandb" to "tensorboard"
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=self.tokenizer,
            args=training_args,
            data_collator=data_collator,
        )

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
            repo_name = f"finetuned-llama2-{int(time.time())}"
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
                    model_name=self.model_name,
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
    miner = Llama2TrainingMiner(
        model_name='NousResearch/Llama-2-7b-chat-hf',
        dataset_id='mlabonne/guanaco-llama2-1k',
        epochs=1,
        batch_size=2,
        learning_rate=2e-5,
        device='cuda',
        hf_token="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",
        central_repo="Tobius/yogpt_test",
    )
    
    async def main():
        try:
            await miner.run_training_loop()
        except KeyboardInterrupt:
            bt.logging.info("Miner stopped.")

    asyncio.run(main())