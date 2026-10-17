from huggingface_hub import hf_hub_download

print("Downloading AtomBlock YOLO dataset...")

file_path = hf_hub_download(
    repo_id="ZhihaoNan/AtomBlock-WebUI",
    repo_type="dataset",
    filename="yolo-dataset.tar",
    local_dir=r"C:\QASense-AI\datasets\atomblock-webui"
)

print("Download completed:")
print(file_path)