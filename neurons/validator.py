import time
import json
import requests
import torch
import bittensor as bt
from template.base.validator import BaseValidatorNeuron
from template.protocol import TrainingProtocol
import asyncio
import os
from dotenv import load_dotenv
import logging
from huggingface_hub import HfApi
from typing import List, Dict, Optional

load_dotenv()  # Load environment variables from .env file

class TrainingValidator(BaseValidatorNeuron):
    def __init__(self, config=None, repo_name=None):
        super().__init__(config=config)
        
        # Load environment variables
        load_dotenv()
        
        # Get the Hugging Face token from the environment variable
        self.hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not self.hf_token:
            bt.logging.error("HUGGING_FACE_TOKEN not found in environment variables")
        
        self.central_repo = os.getenv('CENTRAL_REPO')  # Central repository URL
        if not self.central_repo:
            bt.logging.error("CENTRAL_REPO not found in environment variables")
        
        self.repo_name = repo_name
        self.api = HfApi()

    def fetch_training_metrics_commits(self, repo_id: str, token: Optional[str] = None) -> List[Dict]:
        """
        Fetch commits from a Hugging Face repository that contain training metrics JSON files
        and have 'miner_uid' in the metrics.
        
        Args:
            repo_id (str): The repository ID in format 'username/repository'
            token (str, optional): HuggingFace API token for private repositories.
            
        Returns:
            List[Dict]: List of commits containing valid training metrics.
        """
        commits_with_metrics = []

        try:
            # Fetch the repository files
            files = self.api.list_repo_files(repo_id=repo_id, token=token)
            
            # Check if 'metrics.json' exists in the repository
            if 'metrics.json' in files:
                # Fetch the metrics file
                metrics_file = self.api.download_file('metrics.json', repo_id=repo_id, token=token)
                metrics_data = json.loads(metrics_file)

                # Check for 'miner_uid' in the metrics
                if 'miner_uid' in metrics_data:
                    commits_with_metrics.append({
                        "commit_id": "latest",  # Placeholder for commit ID
                        "miner_uid": metrics_data['miner_uid'],
                        "metrics": metrics_data
                    })
        except Exception as e:
            logging.error(f"Failed to fetch training metrics commits: {str(e)}")

        return commits_with_metrics

    def read_commits(self):
        """Read commits from the central Hugging Face repository."""
        commits = self.fetch_training_metrics_commits(repo_id=self.repo_name, token=self.hf_token)  # Updated method call
        return commits

    def group_commits(self, commits):
        """Group commits by job."""
        job_groups = {}
        for commit in commits:
            job_id = commit.get("metrics", {}).get("job_id")
            if job_id not in job_groups:
                job_groups[job_id] = []
            job_groups[job_id].append(commit)
        return job_groups

    def load_and_evaluate(self, job_groups):
        """Load and evaluate each job."""
        for job_id, commits in job_groups.items():
            metrics_list = self.extract_metrics_by_job_id(commits, job_id)
            if metrics_list:
                self.score_miners(metrics_list, job_id)
                self.mark_job_as_done(job_id)

    def verify_losses(self, metrics_list):
        """Assess and verify the losses for each task."""
        for metric in metrics_list:
            if not self.is_loss_verified(metric):
                continue  # Skip if loss is not verified
            self.update_scores(torch.tensor([1.0]), [metric["miner_uid"]])

    def reward_best_miner(self, job_id):
        """Reward the miner with the best performance for each job."""
        metrics_list = self.get_metrics_for_job(job_id)
        if metrics_list:
            lowest_loss_commit = min(metrics_list, key=lambda x: x["final_loss"])
            self.update_scores(torch.tensor([1.0]), [lowest_loss_commit["miner_uid"]])

    def mark_job_as_done(self, job_id):
        """Mark the evaluated job as complete."""
        self.update_job_status(job_id, status="done")

    def filter_jobs(self, job_groups):
        """Filter and process only unscored jobs."""
        unscored_jobs = {job_id: commits for job_id, commits in job_groups.items() if not self.is_job_scored(job_id)}
        return unscored_jobs

    def evaluate_jobs(self, job_groups):
        """Main method to evaluate jobs."""
        self.load_and_evaluate(job_groups)

    async def forward(self):
        """Main execution method."""
        try:
            logging.info("Fetching commits...")
            commits = self.read_commits()  # This now uses the updated read_commits method
            job_groups = self.group_commits(commits)
            self.evaluate_jobs(job_groups)
        except Exception as e:
            logging.error(f"Error in forward: {str(e)}")

    async def __aenter__(self):
        await self.setup()  # Assuming you have a setup method
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()  # Assuming you have a cleanup method

    async def setup(self):
        # Initialization logic here
        pass

    async def cleanup(self):
        # Cleanup logic here
        pass

    def group_commits_by_job(self, commits):
        """Group commits by job."""
        job_groups = {}
        for commit in commits:
            job_id = commit.get("metrics", {}).get("job_id")
            if job_id not in job_groups:
                job_groups[job_id] = []
            job_groups[job_id].append(commit)
        return job_groups

    def fetch_commits(self):
        """Fetch commits from the Hugging Face Hub repository."""
        try:
            commits = self.api.list_commits(repo_id=self.repo_name)
            return commits
        except Exception as e:
            logging.error(f"Failed to fetch commits: {str(e)}")
            return []

# Main execution
if __name__ == "__main__":
    async def main():
        async with TrainingValidator(repo_name="Tobius/yogpt_test") as validator:
            while True:
                bt.logging.info(f"Validator running... {time.time()}")
                await validator.forward()  # Ensure async function is called properly
                await asyncio.sleep(5)  # Use asyncio.sleep instead of time.sleep

    asyncio.run(main())  # Run the main async function
