import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, auc, 
    roc_curve, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, average_precision_score, precision_recall_curve
)
import os

# Import functions from your src modules
from src.preprocess import parse_seq, tokenize_seq, pad_sequences, extract_go_annotations, preprocess_data
from src.model import GlobalAttention, create_model

# Configuration
TEST_DATA_PATH = 'C:/Users/smm66/Downloads/SIP_BERT/Dataset/Human/Human.test.csv'
GO_ANNOTATION_PATH = 'C:/Users/smm66/Downloads/SIP_BERT/Dataset/Human/GO_annotation_Human.csv'
MODEL_PATH = 'C:/Users/smm66/Downloads/SIP_BERT/Output/Human/fold1/best_model.keras' # adjust the path accordingly
FIXED_LENGTH = 256
OUTPUT_DIR = 'results'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load and prepare data
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    uniref_df = pd.read_csv(GO_ANNOTATION_PATH)
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
    
    # Preprocess data
    test_meta, test_seqs, test_annotations, test_labels = preprocess_data(
        test_df, uniref_df, FIXED_LENGTH
    )
    
    # Load model
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'GlobalAttention': GlobalAttention}
    )
    
    # Make predictions
    test_probabilities = model.predict([test_seqs, test_annotations])
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(test_labels, test_probabilities)
    roc_auc = auc(fpr, tpr)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    
    # Save subsequence-level results
    test_results = test_meta.copy()
    test_results['Pred_label'] = test_predictions
    test_results['Y_Prob'] = test_probabilities
    test_results.to_csv(os.path.join(OUTPUT_DIR, 'Human_result.csv'), index=False)
    
    # Evaluate subsequence-level performance
    print("\nSubsequence-Level Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions))
    print(f"AUC Score: {roc_auc_score(test_labels, test_probabilities):.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', marker='o', label='Optimal Threshold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'subsequence_roc.png'))
    plt.close()
    
    # Protein-level evaluation (Majority Voting)
    agg_df = test_results.groupby('Protein').agg(
        label=('label', 'first'),
        Pred_label=('Pred_label', lambda x: x.mode().iloc[0]),
        Y_Prob=('Y_Prob', 'mean')
    ).reset_index()
    
    print("\nProtein-Level Evaluation (Majority Voting):")
    evaluate_predictions(agg_df['label'], agg_df['Pred_label'], agg_df['Y_Prob'], 
                         os.path.join(OUTPUT_DIR, 'majority_voting_metrics.txt'))
    
    # Protein-level evaluation (One-Hit Rule)
    seq_level_df = test_results.groupby('Protein').agg({
        'label': 'first',
        'Pred_label': lambda x: 1 if 1 in x.values else 0,
        'Y_Prob': 'mean'
    }).reset_index()
    
    print("\nProtein-Level Evaluation (One-Hit Rule):")
    evaluate_predictions(seq_level_df['label'], seq_level_df['Pred_label'], seq_level_df['Y_Prob'],
                         os.path.join(OUTPUT_DIR, 'one_hit_rule_metrics.txt'))
    
    # Save protein-level predictions
    agg_df.to_csv(os.path.join(OUTPUT_DIR, 'seq_level_predictions.csv'), index=False)
    seq_level_df.to_csv(os.path.join(OUTPUT_DIR, 'seq_level_predictions_OH.csv'), index=False)

def evaluate_predictions(y_true, y_pred, y_prob, output_path=None):
    """Evaluate predictions and save metrics"""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"AUC-ROC: {auc_roc:.4f}\n")
            f.write(f"AUPRC: {auprc:.4f}\n")
            f.write(f"MCC: {mcc:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path.replace('.txt', '_roc.png') if output_path else None)
    plt.close()

if __name__ == "__main__":
    main()