import os
from datasets import load_dataset

OUT_DIR = "data/raw/beans"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("beans")  # train/validation/test splits

    # Save as arrow (native HF format) for reproducibility
    ds.save_to_disk(OUT_DIR)
    print("✅ Saved dataset to:", OUT_DIR)
    print(ds)

if __name__ == "__main__":
    main()
