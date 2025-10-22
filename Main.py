import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===================== DATA HANDLING =====================

class VideoDataset(Dataset):
    """Custom dataset for video sequences with temporal context."""
    
    def __init__(self, video_paths, labels, sequence_length=30, transform=None, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        # Apply transformations
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        
        return frames, label
    
    def load_video_frames(self, video_path):
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly across the video
        sample_indices = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count in sample_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            
            frame_count += 1
            
            if len(frames) == self.sequence_length:
                break
        
        cap.release()
        
        # Pad with last frame if necessary
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames

# ===================== MODEL ARCHITECTURE =====================

class SpatialFeatureExtractor(nn.Module):
    """CNN-based spatial feature extractor using pre-trained ResNet."""
    
    def __init__(self, feature_dim=512, pretrained=True):
        super(SpatialFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        features = self.features(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        projected = self.projection(flattened)
        return projected, features

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for frame-level importance weighting."""
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch, sequence, hidden_dim)
        attention_weights = self.attention(lstm_outputs)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        attended = torch.sum(lstm_outputs * attention_weights, dim=1)
        
        return attended, attention_weights

class SpatialAttention(nn.Module):
    """Spatial attention for focusing on relevant regions within frames."""
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, features):
        # Calculate channel-wise statistics
        avg_pool = torch.mean(features, dim=1, keepdim=True)
        max_pool, _ = torch.max(features, dim=1, keepdim=True)
        
        # Concatenate and generate attention map
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = torch.sigmoid(self.conv(concat))
        
        return features * attention_map

class VideoTemporalContextModel(nn.Module):
    """Complete model for video analysis with temporal contextualization."""
    
    def __init__(self, num_classes, sequence_length=30, feature_dim=512, 
                 lstm_hidden=256, lstm_layers=2):
        super(VideoTemporalContextModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # Spatial feature extraction
        self.spatial_extractor = SpatialFeatureExtractor(feature_dim)
        self.spatial_attention = SpatialAttention()
        
        # Temporal modeling with bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(lstm_hidden * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, sequence, channels, height, width)
        batch_size = x.size(0)
        sequence_length = x.size(1)
        
        # Process each frame through spatial feature extractor
        frame_features = []
        spatial_features = []
        
        for t in range(sequence_length):
            frame = x[:, t, :, :, :]
            feat, spatial_feat = self.spatial_extractor(frame)
            
            # Apply spatial attention
            spatial_feat = self.spatial_attention(spatial_feat)
            spatial_features.append(spatial_feat)
            
            # Global pooling after attention
            attended_feat = F.adaptive_avg_pool2d(spatial_feat, (1, 1))
            attended_feat = attended_feat.view(batch_size, -1)
            
            # Combine with projected features
            combined = feat + self.spatial_extractor.projection(attended_feat)
            frame_features.append(combined)
        
        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)
        
        # Temporal modeling with LSTM
        lstm_out, _ = self.lstm(frame_features)
        
        # Apply temporal attention
        attended_temporal, attention_weights = self.temporal_attention(lstm_out)
        
        # Classification
        output = self.classifier(attended_temporal)
        
        return output, attention_weights

# ===================== TRAINING UTILITIES =====================

class VideoModelTrainer:
    """Trainer class for the video temporal context model."""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with different learning rates for pre-trained and new layers
        pretrained_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if 'spatial_extractor.features' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': learning_rate * 0.1},
            {'params': new_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc='Training')
        for batch_idx, (videos, labels) in enumerate(progress_bar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs, _ = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(dataloader, desc='Validation'):
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                outputs, _ = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1, all_labels, all_predictions
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """Complete training loop."""
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, precision, recall, f1, _, _ = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            
            # Model checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_video_model.pth')
                print(f'Model saved with validation accuracy: {val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        return self.history

# ===================== EVALUATION UTILITIES =====================

def evaluate_model(model, test_loader, device, num_classes):
    """Comprehensive model evaluation."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc='Testing'):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs, attention_weights = model(videos)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Print results
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'attention_weights': np.concatenate(all_attention_weights)
    }

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualize_attention_weights(attention_weights, sample_idx=0):
    """Visualize temporal attention weights for a sample."""
    weights = attention_weights[sample_idx].squeeze()
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(weights)), weights)
    plt.xlabel('Frame Index')
    plt.ylabel('Attention Weight')
    plt.title(f'Temporal Attention Weights for Sample {sample_idx}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'attention_weights_sample_{sample_idx}.png')
    plt.show()

# ===================== DATA AUGMENTATION =====================

def get_video_transforms(training=True):
    """Get video frame transformations."""
    if training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ===================== MAIN TRAINING PIPELINE =====================

def main():
    """Main training and evaluation pipeline."""
    
    # Configuration
    config = {
        'num_classes': 10,  # Adjust based on your dataset
        'sequence_length': 30,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'feature_dim': 512,
        'lstm_hidden': 256,
        'lstm_layers': 2,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Note: Replace these with actual video paths and labels from your dataset
    # Example structure:
    train_videos = []  # List of video file paths
    train_labels = []  # Corresponding labels
    val_videos = []
    val_labels = []
    test_videos = []
    test_labels = []
    
    # Create datasets
    train_dataset = VideoDataset(
        train_videos, train_labels,
        sequence_length=config['sequence_length'],
        transform=get_video_transforms(training=True),
        augment=True
    )
    
    val_dataset = VideoDataset(
        val_videos, val_labels,
        sequence_length=config['sequence_length'],
        transform=get_video_transforms(training=False),
        augment=False
    )
    
    test_dataset = VideoDataset(
        test_videos, test_labels,
        sequence_length=config['sequence_length'],
        transform=get_video_transforms(training=False),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Initialize model
    model = VideoTemporalContextModel(
        num_classes=config['num_classes'],
        sequence_length=config['sequence_length'],
        feature_dim=config['feature_dim'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers']
    )
    
    print(f"Model architecture:\n{model}\n")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Initialize trainer
    trainer = VideoModelTrainer(
        model, config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train the model
    print("Starting training...")
    history = trainer.train(
        train_loader, val_loader,
        epochs=config['epochs'],
        patience=10
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_video_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, config['device'], config['num_classes'])
    
    # Visualize results
    plot_training_history(history)
    
    # Generate class names (replace with actual class names)
    class_names = [f'Class_{i}' for i in range(config['num_classes'])]
    plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Visualize attention weights
    visualize_attention_weights(results['attention_weights'])
    
    # Ablation study
    print("\n" + "="*50)
    print("ABLATION STUDY")
    print("="*50)
    ablation_study(config, train_loader, val_loader, test_loader)
    
    return model, results

def ablation_study(config, train_loader, val_loader, test_loader):
    """Perform ablation study to understand component contributions."""
    
    ablation_configs = [
        ("Full Model", {}),
        ("Without Temporal Attention", {'use_temporal_attention': False}),
        ("Without Spatial Attention", {'use_spatial_attention': False}),
        ("Without Pre-training", {'pretrained': False}),
        ("Single LSTM Layer", {'lstm_layers': 1}),
    ]
    
    for name, modifications in ablation_configs:
        print(f"\nTesting: {name}")
        print("-" * 30)
        
        # Create modified config
        ablation_config = config.copy()
        ablation_config.update(modifications)
        
        # Note: You would need to modify the model architecture to support these ablations
        # This is a placeholder for the ablation study structure
        
        # Train and evaluate modified model
        # ... (implementation details)
        
        print(f"Results for {name}: [Placeholder for actual results]")

if __name__ == "__main__":
    model, results = main()
