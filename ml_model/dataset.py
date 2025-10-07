import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
dataset_dir = os.getenv("DATASET_DIR", "./datasets")  # fallback to ./datasets

def load_all_datasets() -> dict:
    """
    Loads ALL datasets in a given folder (including subfolders).
    Automatically detects file type (CSV, TSV, JSON).
    
    Returns:
        dict: {relative_path_without_ext: DataFrame with ['text', 'label']}
    """
    datasets = {}
    
    if not os.path.exists(dataset_dir):
        print(f"⚠️  Dataset directory '{dataset_dir}' not found!")
        return datasets

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            path = os.path.join(root, file)
            name, ext = os.path.splitext(os.path.relpath(path, dataset_dir))

            try:
                if ext == ".tsv":
                    df = pd.read_csv(path, sep="\t", header=None)
                    if df.shape[1] >= 2:
                        df = df[[1, 2]].rename(columns={1: "label", 2: "text"})
                    else:
                        print(f"⚠️  Skipping {file}: Not enough columns")
                        continue
                        
                elif ext == ".csv":
                    df = pd.read_csv(path)
                    if not {"label", "text"}.issubset(df.columns):
                        print(f"⚠️  Skipping {file}: Missing 'label' or 'text' columns")
                        continue
                        
                elif ext == ".json":
                    df = pd.read_json(path)
                    if not {"label", "text"}.issubset(df.columns):
                        print(f"⚠️  Skipping {file}: Missing 'label' or 'text' columns")
                        continue
                else:
                    continue

                # Clean data
                df = df.dropna(subset=["text", "label"])
                df["text"] = df["text"].astype(str)
                
                if len(df) > 0:
                    datasets[name] = df
                    print(f"✅ Loaded {name}: {len(df)} samples")
                    
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                continue

    return datasets