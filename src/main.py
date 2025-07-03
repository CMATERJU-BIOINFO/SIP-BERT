import numpy as np
import pandas as pd
from src.preprocess import preprocess_data, n_tokens
from src.train_eval import run_fold_training
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Constants
FIXED_LENGTH = 256
DATA_PATH = '../SIP_BERT/Dataset/Yeast'
UNIREF_PATH = '../SIP_BERT/Dataset/Yeast/GO_annotation.csv'
TEST_PATH = '../SIP_BERT/Dataset/Yeast/Yeast.test.csv'
OUTPUT_DIR = "../SIP_BERT/Output"  # Set the paths as required

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load UniRef data
uniref_df = pd.read_csv(UNIREF_PATH)
uniref_df['GO_annotation'] = uniref_df.apply(
    lambda row: np.array([
        row['Gene Ontology (biological process) Count'],
        row['Gene Ontology (molecular function) Count'],
        row['Gene Ontology (cellular component) Count']
    ]), axis=1
)
uniref_df = uniref_df.drop(columns=[
    'Gene Ontology (biological process) Count',
    'Gene Ontology (molecular function) Count',
    'Gene Ontology (cellular component) Count'
], errors='ignore')
uniref_df = uniref_df.rename(columns={'Entry Name': 'Protein'})

# Load test data
test_df = pd.read_csv(TEST_PATH)
test_meta, test_seqs, test_annotations, test_labels = preprocess_data(
    test_df, uniref_df, FIXED_LENGTH
)

# Run training for all 11 folds
all_results = []
for fold_idx in range(1, 12):
    fold_result = run_fold_training(
        fold_idx=fold_idx,
        data_path=DATA_PATH,
        uniref_df=uniref_df,
        fixed_length=FIXED_LENGTH,
        n_tokens=n_tokens,
        output_dir=OUTPUT_DIR
    )
    all_results.append(fold_result)

# Evaluate each model on test set
test_roc_aucs = []
plt.figure(figsize=(10, 8))

for result in all_results:
    fold_idx = result['fold']
    model = result['model']
    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold{fold_idx}")
    threshold = result['val_metrics']['optimal_threshold']
    
    # Predict on test set
    test_probabilities = model.predict([test_seqs, test_annotations])
    test_predictions = (test_probabilities >= threshold).astype(int)
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(test_labels, test_probabilities)
    roc_auc = auc(fpr, tpr)
    test_roc_aucs.append(roc_auc)
    
    # Save test predictions
    test_result_df = test_meta.copy()
    test_result_df['Pred_label'] = test_predictions
    test_result_df['Y_Prob'] = test_probabilities
    test_result_df.to_csv(os.path.join(fold_output_dir, f'test_results_fold{fold_idx}.csv'), index=False)
    
    # Save test metrics
    with open(os.path.join(fold_output_dir, f'test_metrics_fold{fold_idx}.txt'), 'w') as f:
        f.write(f"Test ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Optimal Threshold: {threshold:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_labels, test_predictions)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(test_labels, test_predictions))
    
    # Add to ROC plot
    plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {fold_idx} (AUC = {roc_auc:.4f})')

# Save combined ROC plot
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Folds (Test Set)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, 'all_folds_test_roc.png'))
plt.close()

# Save final results summary
final_results = pd.DataFrame({
    'Fold': list(range(1, 12)),
    'ROC_AUC': test_roc_aucs
})
final_results.to_csv(os.path.join(OUTPUT_DIR, 'final_fold_results.csv'), index=False)

# Print summary
print("\nFinal Results Summary:")
print(f"Average Test ROC-AUC: {np.mean(test_roc_aucs):.4f}")
print(f"Standard Deviation: {np.std(test_roc_aucs):.4f}")
with open(os.path.join(OUTPUT_DIR, 'final_summary.txt'), 'w') as f:
    f.write(f"Average Test ROC-AUC: {np.mean(test_roc_aucs):.4f}\n")
    f.write(f"Standard Deviation: {np.std(test_roc_aucs):.4f}\n")
    f.write("\nIndividual Fold Results:\n")
    f.write(final_results.to_string(index=False))