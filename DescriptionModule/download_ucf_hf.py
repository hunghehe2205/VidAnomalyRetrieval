#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
token = os.environ["HF_READ_TOKEN"]

snapshot_download(repo_id="hungnghehe/UCFClipFeature", local_dir="UCFClipFeature", token=token)
snapshot_download(repo_id="hungnghehe/Org_vadclip", local_dir="Org_vadclip", token=token)
