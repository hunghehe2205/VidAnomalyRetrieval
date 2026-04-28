from huggingface_hub import snapshot_download

hf_token = ""
# dataset
snapshot_download(
    repo_id="hungnghehe/UCFClipFeature",
    repo_type="dataset",
    local_dir="UCFClipFeature",
    token=hf_token,
)

# model
snapshot_download(
    repo_id="hungnghehe/Org_vadclip",
    local_dir="Org_vadclip",   # mặc định là model
    token=hf_token,
)

# model
snapshot_download(
    repo_id="hungnghehe/Holmes-VAU-ATS",
    local_dir="Holmes-VAU-ATS",   # mặc định là model
    token=hf_token,
)
