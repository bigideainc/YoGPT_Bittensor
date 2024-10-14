import json
from huggingface_hub import HfApi, Repository
import os
from datetime import datetime

def commit_to_central_repo(hf_token: str, central_repo: str, model_repo: str, metrics: dict):
    """
    Commit metrics and model information to a central Hugging Face repository.
    
    :param hf_token: Hugging Face API token
    :param central_repo: Name of the central repository to commit to
    :param model_repo: URL of the model repository
    :param metrics: Dictionary containing training metrics
    :return: URL of the committed file in the central repository
    """
    api = HfApi(token=hf_token)
    
    # Ensure the central repository exists, create if it doesn't
    try:
        api.repo_info(repo_id=central_repo)
    except Exception:
        api.create_repo(repo_id=central_repo, private=True)
    
    # Clone the central repository
    repo = Repository(local_dir="central_repo", clone_from=central_repo, use_auth_token=hf_token)
    repo.git_pull()

    # Create a unique filename for this commit
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_run_{timestamp}.json"

    # Prepare the data to be committed
    data = {
        "model_repo": model_repo,
        "metrics": metrics,
        "timestamp": timestamp
    }

    # Write the data to a file
    with open(os.path.join("central_repo", filename), "w") as f:
        json.dump(data, f, indent=2)

    # Commit and push the changes
    repo.push_to_hub(commit_message=f"Training run {timestamp}")

    # Get the URL of the committed file
    file_url = f"https://huggingface.co/{central_repo}/blob/main/{filename}"

    return file_url