from preprocess import preprocess_data
from model import GlobalAttention, create_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import os

def run_fold_training(fold_idx, data_path, uniref_df, fixed_length, n_tokens, output_dir):
    print(f"\nProcessing fold {fold_idx}...")
    
    # Create fold-specific output directory
    fold_output_dir = os.path.join(output_dir, f"fold{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Load data
    train_file = os.path.join(data_path, f"Yeast_{fold_idx}.train.csv")
    valid_file = os.path.join(data_path, f"Yeast_{fold_idx}.valid.csv")
    
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)

    # Preprocess data
    train_meta, train_seqs, train_annotations, train_labels = preprocess_data(train_df, uniref_df, fixed_length)
    valid_meta, valid_seqs, valid_annotations, valid_labels = preprocess_data(valid_df, uniref_df, fixed_length)
    
    # Build model
    model = create_model(fixed_length, n_tokens, 3)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(fold_output_dir, f'best_protein_seq_model{fold_idx}.keras'), 
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )

    # Train model
    history = model.fit(
        [train_seqs, train_annotations], train_labels,
        epochs=150,
        batch_size=32,
        validation_data=([valid_seqs, valid_annotations], valid_labels),
        callbacks=[reduce_lr, checkpoint]
    )
    
    # Save training history plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy - Fold {fold_idx}')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss - Fold {fold_idx}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(fold_output_dir, f'training_history_fold{fold_idx}.png'))
    plt.close()
    
    # Load best model
    best_model = keras.models.load_model(
        os.path.join(fold_output_dir, f'best_protein_seq_model{fold_idx}.keras'), 
        custom_objects={'GlobalAttention': GlobalAttention}
    )
    
    # Validate
    val_probabilities = best_model.predict([valid_seqs, valid_annotations])
    fpr, tpr, thresholds = roc_curve(valid_labels, val_probabilities)
    roc_auc = auc(fpr, tpr)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    val_predictions = (val_probabilities >= optimal_threshold).astype(int)
    
    # Save ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', label=f'Threshold={optimal_threshold:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_idx}')
    plt.legend()
    plt.savefig(os.path.join(fold_output_dir, f'roc_curve_fold{fold_idx}.png'))
    plt.close()
    
    # Save PR curve
    precision, recall, _ = precision_recall_curve(valid_labels, val_probabilities)
    pr_auc = average_precision_score(valid_labels, val_probabilities)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, lw=2, color='darkorange', label=f'PR-AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold {fold_idx}')
    plt.legend(loc="lower left")
    plt.grid(False)
    plt.savefig(os.path.join(fold_output_dir, f'pr_curve_fold{fold_idx}.png'))
    plt.close()
    
    # Save validation metrics
    with open(os.path.join(fold_output_dir, f'val_metrics_fold{fold_idx}.txt'), 'w') as f:
        f.write(f"Validation ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(valid_labels, val_predictions)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(valid_labels, val_predictions))
    
    return {
        'fold': fold_idx,
        'model': best_model,
        'val_metrics': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': optimal_threshold
        }
    }