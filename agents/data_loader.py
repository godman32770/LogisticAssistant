import pandas as pd
import os
from core import RAGState


def load_inventory_data(state: RAGState) -> RAGState:
    csv_path = "data/DataDictionary.csv"
    print("load_inventory_data called")
    print("State before loading:", list(state.keys()))
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Inventory file not found: {csv_path}")
    
    try:
        inventory_df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if "data" not in state:
        state["data"] = {}
        
    state["data"]["inventory"] = inventory_df
    
    print("State after loading:", list(state.keys()))
    print("State object ID:", id(state))
    print("Inventory keys:", state["data"]["inventory"].columns.tolist())

    return state

