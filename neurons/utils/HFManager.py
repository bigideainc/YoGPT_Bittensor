import json
from huggingface_hub import HfApi, Repository
import os
from datetime import datetime

def commit_to_central_repo(hf_token: str, central_repo: str, model_repo: str, metrics: dict, miner_uid: int):
    """
    Upload metrics and model information to a central Hugging Face repository using the Hub API.
    
    :param hf_token: Hugging Face API token
    :param central_repo: Name of the central repository to upload to
    :param model_repo: URL of the model repository
    :param metrics: Dictionary containing training metrics
    :return: URL of the uploaded file in the central repository
    """
    api = HfApi(token=hf_token)
    
    # Ensure the central repository exists, create if it doesn't
    try:
        api.repo_info(repo_id=central_repo)
    except Exception:
        api.create_repo(repo_id=central_repo, private=True)
    
    # Create a unique filename for this upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_run_{timestamp}.json"

    # Prepare the data to be uploaded
    data = {
        "model_repo": model_repo,
        "metrics": metrics,
        "timestamp": timestamp,
        "miner_uid": miner_uid
    }

    # Write the data to a temporary file
    temp_file_path = f"/tmp/{filename}"
    with open(temp_file_path, "w") as f:
        json.dump(data, f, indent=2)

    # Upload the file to the repository
    api.upload_file(
        path_or_fileobj=temp_file_path,
        path_in_repo=filename,
        repo_id=central_repo,
        commit_message=f"Training run {timestamp}"
    )

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Get the URL of the uploaded file
    file_url = f"https://huggingface.co/{central_repo}/blob/main/{filename}"

    return file_url