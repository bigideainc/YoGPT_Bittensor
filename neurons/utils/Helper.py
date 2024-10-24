import asyncio
import json
import os
import sys
import time
import aiohttp
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository,hf_hub_download
from datetime import datetime
from typing import List, Dict, Optional
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
TOKEN = os.getenv("TOKEN")
             
async def update_job_status(job_id, status):
    url = f"{BASE_URL}/update-status/{job_id}"
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}
        async with session.patch(url, json={'status': status}, headers=headers) as response:
            try:
                if response.status == 200:
                    print(f"Status updated to {status} for job {job_id}")
                else:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as err:
                print(f"Failed to update status for job {job_id}: {err}")
            except Exception as e:
                print(f"An error occurred: {e}")
                                
async def register_completed_job(job_id, huggingFaceRepoId, loss, accuracy, total_pipeline_time,miner_uid):
    url = f"{BASE_URL}/complete-training"
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}
        payload = {
            'jobId': job_id,
            'huggingFaceRepoId': huggingFaceRepoId,
            'loss': loss,
            "accuracy": accuracy,
            'minerId': miner_uid,
            'totalPipelineTime': total_pipeline_time
        }
        async with session.post(url, json=payload, headers=headers) as response:
            try:
                if response.status == 200:
                    print(f"Completed job registered successfully for job {job_id}")
                else:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as err:
                print(f"Failed to register completed job {job_id}: {err}")
            except Exception as e:
                print(f"An error occurred: {e}")