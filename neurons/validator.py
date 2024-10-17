import time
import json
import requests
import torch
import bittensor as bt
from template.base.validator import BaseValidatorNeuron
from template.protocol import TrainingProtocol

class TrainingValidator(BaseValidatorNeuron):
    def __init__(self, config=None, central_repo: str = "Tobius/yogpt_test"):
        super().__init__(config=config)
        
        self.central_repo = central_repo  # Central repository URL

    def fetch_commits(self):
        """Fetch commits from the central repository."""
        try:
            # Replace with the actual API endpoint or method to retrieve commits.
            response = requests.get(f"https://api.example.com/repos/{self.central_repo}/commits")
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()  # Assuming the response is in JSON format

        except Exception as e:
            bt.logging.error(f"Error fetching commits: {str(e)}")
            return []

    def extract_unique_job_ids(self, commits):
        """Extract unique job_ids from the list of commits."""
        job_ids = set()
        
        for commit in commits:
            job_id = commit.get("metrics", {}).get("job_id")
            if job_id:
                job_ids.add(job_id)
        
        return list(job_ids)

    def extract_metrics_by_job_id(self, commits, job_id):
        """Extract metrics for the given job_id from the commits."""
        metrics_list = []
        
        for commit in commits:
            if commit.get("metrics") and commit["metrics"].get("job_id") == job_id:
                metrics_list.append({
                    "final_loss": commit["metrics"]["final_loss"],
                    "miner_uid": commit["miner_uid"],
                    "model_repo": commit["model_repo"]
                })

        return metrics_list

    def score_miners(self, metrics_list, job_id):
        """Score miners based on the lowest loss for a specific job_id."""
        if not metrics_list:
            bt.logging.info(f"No miners to score for job_id {job_id}.")
            return

        # Sort metrics by final_loss and get the miner with the lowest loss
        lowest_loss_commit = min(metrics_list, key=lambda x: x["final_loss"])
        bt.logging.info(f"Scoring miner {lowest_loss_commit['miner_uid']} with loss {lowest_loss_commit['final_loss']} for job_id {job_id}")

        # Update scores for the miner with the lowest loss
        self.update_scores(torch.tensor([1.0]), [lowest_loss_commit["miner_uid"]])

    async def forward(self):
        """Query miners for training and evaluate their performance."""
        try:
            # Fetch commits from the central repo
            commits = self.fetch_commits()
            
            # Extract unique job IDs
            unique_job_ids = self.extract_unique_job_ids(commits)
            
            # Print unique job IDs
            bt.logging.info(f"Unique job_ids found: {unique_job_ids}")
            
            # Iterate over each job_id and score miners
            for job_id in unique_job_ids:
                bt.logging.info(f"Starting scoring for job_id: {job_id}")
                
                # Extract metrics for the current job_id
                metrics_list = self.extract_metrics_by_job_id(commits, job_id)
                
                # Score miners based on extracted metrics
                self.score_miners(metrics_list, job_id)
                
                bt.logging.info(f"Finished scoring for job_id: {job_id}")

        except Exception as e:
            bt.logging.error(f"Error in forward: {str(e)}")

# Main execution
if __name__ == "__main__":
    with TrainingValidator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            await validator.forward()  # Ensure async function is called properly
            time.sleep(5)
