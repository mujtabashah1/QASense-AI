from datasets import load_dataset

print("Downloading WebUI dataset...")

dataset = load_dataset(
    "ronantakizawa/webui"
)

dataset.save_to_disk(
    "./datasets/webui"
)

print("WebUI downloaded successfully.")
print(dataset)