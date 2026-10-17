from huggingface_hub import snapshot_download

print("Starting WebUI download...")

snapshot_download(
    repo_id="ronantakizawa/webui",
    repo_type="dataset",
    local_dir=r"C:\QASense-AI\datasets\webui"
)

print("WebUI download completed.")