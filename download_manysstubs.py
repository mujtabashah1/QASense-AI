from huggingface_hub import snapshot_download

print("Downloading ManySStuBs4J...")

snapshot_download(
    repo_id="claudios/ManySStuBs4J",
    repo_type="dataset",
    local_dir=r"C:\QASense-AI\datasets\manysstubs4j"
)

print("ManySStuBs4J download completed.")