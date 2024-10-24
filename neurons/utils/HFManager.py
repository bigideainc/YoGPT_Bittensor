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


def fetch_training_metrics_commits(repo_id: str) -> List[Dict]:
    try:
        token = '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'
        # Initialize the Hugging Face API client
        api = HfApi(token=token)

        # Construct the model repo URL
        model_repo_url = f"https://huggingface.co/{repo_id}"

        # Fetch all commits for the repository
        commits = api.list_repo_commits(
            repo_id=repo_id,
            token=token
        )

        training_metrics = []
        processed_commits = 0

        print(f"Found {len(commits)} total commits in repository")

        for commit in commits:
            try:
                # Get the list of files in this commit
                files = api.list_repo_tree(
                    repo_id=repo_id,
                    revision=commit.commit_id,
                    token=token
                )

                # Look for JSON files
                json_files = [f for f in files if f.path.endswith('.json')]

                for json_file in json_files:
                    try:
                        # Download the file at the specific commit
                        local_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=json_file.path,
                            revision=commit.commit_id,
                            token=token
                        )

                        # Read and parse the JSON file content
                        with open(local_path, 'r') as f:
                            content = f.read()
                            metrics_data = json.loads(content)

                        # Ensure 'metrics' and 'miner_uid' exist in the JSON data
                        if isinstance(metrics_data, dict) and "metrics" in metrics_data and "miner_uid" in metrics_data:
                            # Create metrics entry
                            metrics_entry = {
                                "model_repo": model_repo_url,
                                "metrics": metrics_data["metrics"],
                                "miner_uid": metrics_data["miner_uid"],
                                "timestamp": metrics_data.get("timestamp", "unknown")
                            }

                            training_metrics.append(metrics_entry)
                            processed_commits += 1

                    except json.JSONDecodeError:
                        print(f"Could not decode JSON in file: {json_file.path}")
                        continue

            except Exception as e:
                print(f"Error processing commit {commit.commit_id}: {str(e)}")
                continue

        print(f"Successfully processed {processed_commits} commits with valid metrics")
        return training_metrics

    except Exception as e:
        print(f"Error fetching commits: {str(e)}")
        return []