"""Configuration file for patent classification."""

class Config:
    # Model configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MAX_LENGTH = 2048
    TEMPERATURE = 0.1
    TOP_P = 0.9
    
    # Dataset configuration
    DATASET_NAME = "ccdv/patent-classification"
    SUBSET = "abstract"  # Use "abstract" subset
    
    # Training configuration
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Class labels mapping
    CLASS_LABELS = {
        0: "Human Necessities",
        1: "Performing Operations; Transporting", 
        2: "Chemistry; Metallurgy",
        3: "Textiles; Paper",
        4: "Fixed Constructions",
        5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
        6: "Physics",
        7: "Electricity",
        8: "General tagging of new or cross-sectional technology"
    }
    
    # Output paths
    OUTPUT_DIR = "./results"
    MODEL_SAVE_DIR = "./saved_models"
    LOGS_DIR = "./logs"