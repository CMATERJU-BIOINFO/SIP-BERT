# SIP-BERT : A multi-organism deep strategy for predicting self interaction in proteins

![WorkflowF](https://github.com/user-attachments/assets/c1921303-f992-4128-90d2-4f4f13532083)


## Project Overview
Self-interacting proteins (SIPs) are critical to cellular regulation, yet their experimental identification remains challenging due to high costs, inefficiencies, and frequent false positives. Leveraging recent advances in deep language models, we introduce SIP-BERT, a family of lightweight transformer-based models trained on organism-specific self-interaction datasets curated from existing protein–protein interaction databases. We developed three variants SIP-BERT(H), SIP-BERT(Y), and SIP-BERT(HY)-trained on human, yeast, and combined datasets, respectively. These models significantly outperform existing methods, exceeding baseline accuracies by 18%, 8% and 15% respectively. SIP-BERT models also generalise effectively to under-annotated organisms such as the mouse and the fruit fly, achieving high recall despite minimal labelled data. Furthermore, structural analysis of predicted false positives using PDB-derived alpha-carbon distance maps reveals close spatial residue proximities, suggesting plausible but undocumented self-interactions. These results highlight the potential of SIP-BERT to uncover novel SIPs and expand our understanding of protein self-interaction across diverse species. 

Benchmark datasets and all SIP-BERT models are available in this repository.

## Input Data Structure
Protein self interaction data have been organized as follows for each species (Human, Yeast, Fruitfly, Mouse):
```
Dataset/{Species}
├── GO_annotation_{species}.csv
├── {species}.test.csv
├── {species}_1.train.csv
├── {species}_1.valid.csv
├── {species}_2.train.csv
├── {species}_2.valid.csv
...
├── {species}_10.train.csv
└── {species}_10.valid.csv
```

### File Descriptions
1. **GO Annotation File** (`GO_annotation.csv`)
   - Contains protein-level Gene Ontology annotations
   - Required columns:
     - `Protein`: Protein identifier
     - `Gene Ontology (biological process) Count`
     - `Gene Ontology (molecular function) Count`
     - `Gene Ontology (cellular component) Count`
     - The go annotation can also be downloaded from [Uniprot ](https://www.uniprot.org/) and then excecute Go_annot_curate.py to get the model required GO_annotation

2. **Test File** (`{species}.test.csv`)
   - Contains the test dataset for final evaluation
   - Required columns:
     - `Protein`: Protein identifier
     - `seq`: Protein sequence
     - `label`: Binary interaction label (0/1)

3. **Training/Validation Files**
   - 10 pairs of training/validation files (fold 1-10)
   - Naming convention: `{species}_{fold_number}.{train/valid}.csv`
   - Same columns as test file

## Installation
bash command
# Clone repository
```bash
git clone https://github.com/CMATERJU-BIOINFO/SIP-BERT.git
cd SIP-BERT
```
# Install dependencies
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```
```
SIP-BERT/
├── Dataset/                # Input data (organized by species)
├── Output/                 # Output directory (created during execution)
├── src/
│   ├── main.py             # Main execution script
│   ├── model.py            # Model architecture implementation
│   ├── preprocess.py       # Data preprocessing utilities
│   └── train_eval.py       # Training and evaluation functions
└── README.md
```


## Execution
1. **Configure paths in src/main.py**
```bash 
    SPECIES = ['Human', 'Yeast', 'Fruitfly', 'Mouse']
    BASE_DATA_PATH = 'data'  # Path to species folders
    OUTPUT_DIR = "results"
    FIXED_LENGTH = 256       # Protein sequence length
    - python src/main.py
```

## Output Structure
Results are organized by species in the output directory:
```
Output/
├── Human/
│   ├── fold1/
│   │   ├── best_model.keras          # Trained model weights
│   │   ├── training_history.png      # Training/validation metrics plot
│   │   ├── roc_curve.png             # Validation ROC curve
│   │   ├── pr_curve.png              # Validation PR curve
│   │   ├── val_metrics.txt           # Validation metrics
│   │   ├── test_results.csv          # Test set predictions
│   │   └── test_metrics.txt          # Test set metrics
│   ├── fold2/
│   │   └── ... (same files as fold1)
│   ├── ... (fold3 to fold10)
│   ├── all_folds_test_roc.png        # Combined test ROC curves
│   ├── final_results.csv             # Aggregated fold performance
│   └── summary.txt                   # Performance summary
├── Yeast/
│   └── ... (same structure as Human)
├── Fruitfly/
│   └── ... (same structure as Human)
└── Mouse/
    └── ... (same structure as Human)
```


## Model Parameters
Key parameters in src/model.py:
```bash
d_hidden_seq = 128          # Sequence embedding dimension
d_hidden_global = 512       # GO annotation embedding dimension
n_blocks = 6                # Number of processing blocks
n_heads = 4                 # Attention heads
d_key = 64                  # Attention key dimension
conv_kernel_size = 9        # Convolution kernel size
wide_conv_dilation_rate = 5 # Dilation rate for wide convolution
dropout_rate = 0.5          # Dropout rate
```
## Results Interpretation
Output files include:

Validation Metrics (per fold):

ROC-AUC: Area under Receiver Operating Characteristic curve

PR-AUC: Area under Precision-Recall curve

Optimal Threshold: Best classification threshold

Confusion Matrix: TP, FP, TN, FN counts

Classification Report: Precision, Recall, F1-score

**Test Results:**

test_results.csv: Predictions with probabilities

test_metrics.txt: Comprehensive test performance

**Aggregate Results:**

final_results.csv: ROC-AUC scores across all folds

summary.txt: Average performance metrics

all_folds_test_roc.png: Combined ROC curves for all folds

**Customization**
Modify these aspects in the code:

Data paths: Update BASE_DATA_PATH in main.py

Model architecture: Adjust parameters in model.py

Training parameters: Modify epochs, batch size in train_eval.py
