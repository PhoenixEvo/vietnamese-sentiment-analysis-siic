import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_plot_lstm_history():
    """Load and plot LSTM training history from pkl file"""
    try:
        # Load LSTM results
        with open('results/lstm_results.pkl', 'rb') as f:
            lstm_results = pickle.load(f)
        
        print("LSTM results keys:", lstm_results.keys())
        print("LSTM results content:")
        for key, value in lstm_results.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"{key}: length {len(value)}, sample: {value[:3] if len(value) > 0 else 'empty'}")
            else:
                print(f"{key}: {value}")
        
        # Check if training history exists
        if 'training_history' in lstm_results:
            plot_lstm_training_history(lstm_results['training_history'])
        elif any(key in lstm_results for key in ['train_losses', 'train_accuracies', 'val_losses', 'val_accuracies']):
            plot_lstm_training_history(lstm_results)
        else:
            print("No training history found in LSTM results")
            
    except FileNotFoundError:
        print("lstm_results.pkl not found")
    except Exception as e:
        print(f"Error loading LSTM results: {e}")

def load_and_plot_phobert_history():
    """Load and plot PhoBERT training history from pkl file"""
    try:
        # Load PhoBERT results
        with open('results/phobert_results_20250716_172440.pkl', 'rb') as f:
            phobert_results = pickle.load(f)
        
        print("\nPhoBERT results keys:", phobert_results.keys())
        print("PhoBERT results content:")
        for key, value in phobert_results.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"{key}: length {len(value)}, sample: {value[:3] if len(value) > 0 else 'empty'}")
            else:
                print(f"{key}: {type(value)}")
        
        # Check if training history exists
        if 'training_history' in phobert_results:
            plot_phobert_training_history(phobert_results['training_history'])
        elif any(key in phobert_results for key in ['train_losses', 'train_accuracies', 'val_losses', 'val_accuracies']):
            plot_phobert_training_history(phobert_results)
        else:
            print("No training history found in PhoBERT results")
            
    except FileNotFoundError:
        print("phobert_results_20250716_172440.pkl not found")
    except Exception as e:
        print(f"Error loading PhoBERT results: {e}")

def plot_lstm_training_history(results):
    """Plot LSTM training history"""
    plt.figure(figsize=(15, 5))
    
    # Extract training history
    train_losses = results.get('train_losses', [])
    train_accuracies = results.get('train_accuracies', [])
    val_losses = results.get('val_losses', [])
    val_accuracies = results.get('val_accuracies', [])
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    if train_losses and val_losses:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.title('LSTM Model - Training & Validation Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    if train_accuracies and val_accuracies:
        epochs = range(1, len(train_accuracies) + 1)
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('LSTM Model - Training & Validation Accuracy', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot summary statistics
    plt.subplot(1, 3, 3)
    if train_losses and val_losses and train_accuracies and val_accuracies:
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        final_train_acc = train_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        best_val_acc = max(val_accuracies)
        
        metrics = ['Final Train\nLoss', 'Final Val\nLoss', 'Final Train\nAcc', 'Final Val\nAcc', 'Best Val\nAcc']
        values = [final_train_loss, final_val_loss, final_train_acc, final_val_acc, best_val_acc]
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('LSTM Model - Final Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/lstm_training_history_recovered.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nLSTM Training Summary:")
    if train_losses and val_losses:
        print(f"- Total epochs: {len(train_losses)}")
        print(f"- Final training loss: {train_losses[-1]:.4f}")
        print(f"- Final validation loss: {val_losses[-1]:.4f}")
        print(f"- Final training accuracy: {train_accuracies[-1]:.4f}")
        print(f"- Final validation accuracy: {val_accuracies[-1]:.4f}")
        print(f"- Best validation accuracy: {max(val_accuracies):.4f}")

def plot_phobert_training_history(results):
    """Plot PhoBERT training history"""
    plt.figure(figsize=(15, 5))
    
    # Handle different possible structures
    if isinstance(results, dict):
        train_losses = results.get('train_losses', results.get('train_loss', []))
        train_accuracies = results.get('train_accuracies', results.get('train_accuracy', []))
        val_losses = results.get('val_losses', results.get('val_loss', []))
        val_accuracies = results.get('val_accuracies', results.get('val_accuracy', []))
    else:
        print("Unexpected PhoBERT results structure")
        return
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    if train_losses and val_losses:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'g-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'orange', label='Validation Loss', linewidth=2)
        plt.title('PhoBERT Model - Training & Validation Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    if train_accuracies and val_accuracies:
        epochs = range(1, len(train_accuracies) + 1)
        plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_accuracies, 'orange', label='Validation Accuracy', linewidth=2)
        plt.title('PhoBERT Model - Training & Validation Accuracy', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot summary statistics
    plt.subplot(1, 3, 3)
    if train_losses and val_losses and train_accuracies and val_accuracies:
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        final_train_acc = train_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        best_val_acc = max(val_accuracies)
        
        metrics = ['Final Train\nLoss', 'Final Val\nLoss', 'Final Train\nAcc', 'Final Val\nAcc', 'Best Val\nAcc']
        values = [final_train_loss, final_val_loss, final_train_acc, final_val_acc, best_val_acc]
        colors = ['lightgreen', 'lightyellow', 'lightblue', 'lightcoral', 'lightpink']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('PhoBERT Model - Final Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/phobert_training_history_recovered.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPhoBERT Training Summary:")
    if train_losses and val_losses:
        print(f"- Total epochs: {len(train_losses)}")
        print(f"- Final training loss: {train_losses[-1]:.4f}")
        print(f"- Final validation loss: {val_losses[-1]:.4f}")
        print(f"- Final training accuracy: {train_accuracies[-1]:.4f}")
        print(f"- Final validation accuracy: {val_accuracies[-1]:.4f}")
        print(f"- Best validation accuracy: {max(val_accuracies):.4f}")

def plot_comparison():
    """Plot comparison between models"""
    try:
        # Load both results
        with open('results/lstm_results.pkl', 'rb') as f:
            lstm_results = pickle.load(f)
        
        with open('results/phobert_results_20250716_172440.pkl', 'rb') as f:
            phobert_results = pickle.load(f)
        
        plt.figure(figsize=(12, 8))
        
        # Extract LSTM data
        lstm_data = {}
        if 'training_history' in lstm_results:
            lstm_data = lstm_results['training_history']
        else:
            lstm_data = lstm_results
            
        # Extract PhoBERT data  
        phobert_data = phobert_results
        
        # Validation accuracy comparison
        plt.subplot(2, 2, 1)
        has_data = False
        if 'val_accuracies' in lstm_data and lstm_data['val_accuracies']:
            lstm_epochs = range(1, len(lstm_data['val_accuracies']) + 1)
            plt.plot(lstm_epochs, lstm_data['val_accuracies'], 'b-', label='LSTM', linewidth=2, marker='o')
            has_data = True
        
        phobert_val_acc = phobert_data.get('val_accuracies', phobert_data.get('val_accuracy', []))
        if phobert_val_acc:
            phobert_epochs = range(1, len(phobert_val_acc) + 1)
            plt.plot(phobert_epochs, phobert_val_acc, 'g-', label='PhoBERT', linewidth=2, marker='s')
            has_data = True
        
        plt.title('Validation Accuracy Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        if has_data:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Validation loss comparison
        plt.subplot(2, 2, 2)
        has_data = False
        if 'val_losses' in lstm_data and lstm_data['val_losses']:
            lstm_epochs = range(1, len(lstm_data['val_losses']) + 1)
            plt.plot(lstm_epochs, lstm_data['val_losses'], 'b-', label='LSTM', linewidth=2, marker='o')
            has_data = True
        
        phobert_val_loss = phobert_data.get('val_losses', phobert_data.get('val_loss', []))
        if phobert_val_loss:
            phobert_epochs = range(1, len(phobert_val_loss) + 1)
            plt.plot(phobert_epochs, phobert_val_loss, 'g-', label='PhoBERT', linewidth=2, marker='s')
            has_data = True
        
        plt.title('Validation Loss Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if has_data:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training accuracy comparison
        plt.subplot(2, 2, 3)
        has_data = False
        if 'train_accuracies' in lstm_data and lstm_data['train_accuracies']:
            lstm_epochs = range(1, len(lstm_data['train_accuracies']) + 1)
            plt.plot(lstm_epochs, lstm_data['train_accuracies'], 'b-', label='LSTM', linewidth=2, marker='o')
            has_data = True
        
        phobert_train_acc = phobert_data.get('train_accuracies', phobert_data.get('train_accuracy', []))
        if phobert_train_acc:
            phobert_epochs = range(1, len(phobert_train_acc) + 1)
            plt.plot(phobert_epochs, phobert_train_acc, 'g-', label='PhoBERT', linewidth=2, marker='s')
            has_data = True
        
        plt.title('Training Accuracy Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        if has_data:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training loss comparison
        plt.subplot(2, 2, 4)
        has_data = False
        if 'train_losses' in lstm_data and lstm_data['train_losses']:
            lstm_epochs = range(1, len(lstm_data['train_losses']) + 1)
            plt.plot(lstm_epochs, lstm_data['train_losses'], 'b-', label='LSTM', linewidth=2, marker='s')
            has_data = True
        
        phobert_train_loss = phobert_data.get('train_losses', phobert_data.get('train_loss', []))
        if phobert_train_loss:
            phobert_epochs = range(1, len(phobert_train_loss) + 1)
            plt.plot(phobert_epochs, phobert_train_loss, 'g-', label='PhoBERT', linewidth=2, marker='s')
            has_data = True
        
        plt.title('Training Loss Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if has_data:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/models_training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print comparison summary
        print("\n📊 Model Comparison Summary:")
        if 'val_accuracies' in lstm_data and lstm_data['val_accuracies']:
            print(f"LSTM - Best Validation Accuracy: {max(lstm_data['val_accuracies']):.4f}")
        
        if phobert_val_acc:
            print(f"PhoBERT - Best Validation Accuracy: {max(phobert_val_acc):.4f}")
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")

if __name__ == "__main__":
    print("=== Loading và Plotting Training History của các Model ===\n")
    
    # Load và plot LSTM history
    print("1. LSTM Training History:")
    load_and_plot_lstm_history()
    
    print("\n" + "="*60 + "\n")
    
    # Load và plot PhoBERT history
    print("2. PhoBERT Training History:")
    load_and_plot_phobert_history()
    
    print("\n" + "="*60 + "\n")
    
    # Plot comparison
    print("3. Model Comparison:")
    plot_comparison()
    
    print("\nĐã tạo các file plot:")
    print("- results/lstm_training_history_recovered.png")
    print("- results/phobert_training_history_recovered.png") 
    print("- results/models_training_comparison.png") 