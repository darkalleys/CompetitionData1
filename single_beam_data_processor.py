import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt

class SingleBeamDataProcessor:
    """单波束数据处理器 - 智能采样策略"""
    
    def __init__(self, n_beams=165, n_pings=60):
        self.n_beams = n_beams
        self.n_pings = n_pings
        
    def extract_beam_sequences(self, multibeam_data, target_info):
        """
        从多波束数据中提取单波束序列
        
        Args:
            multibeam_data: shape (n_pings, n_beams, n_time_samples)
            target_info: 目标信息 [(ping_id, beam_id, target_type), ...]
        
        Returns:
            sequences: 波束序列数据
            labels: 对应标签
            metadata: 序列元数据
        """
        sequences = []
        labels = []
        metadata = []
        
        # 创建目标映射
        target_map = {}
        for ping_id, beam_id, target_type in target_info:
            if ping_id not in target_map:
                target_map[ping_id] = {}
            target_map[ping_id][beam_id] = target_type
        
        # 提取所有波束序列
        for ping_id in range(self.n_pings):
            for beam_id in range(self.n_beams):
                # 提取单波束时间序列
                sequence = multibeam_data[ping_id, beam_id, :]
                
                # 确定标签
                if ping_id in target_map and beam_id in target_map[ping_id]:
                    label = target_map[ping_id][beam_id]  # 1, 0, 或 -1
                else:
                    label = -1  # 无目标
                
                sequences.append(sequence)
                labels.append(label)
                metadata.append({
                    'ping_id': ping_id,
                    'beam_id': beam_id,
                    'has_target': label != -1
                })
        
        return np.array(sequences), np.array(labels), metadata
    
    def intelligent_sampling(self, sequences, labels, metadata, strategy='balanced'):
        """
        智能采样策略
        
        Args:
            strategy: 'balanced', 'all', 'target_focused'
        """
        if strategy == 'all':
            return sequences, labels, metadata
        
        elif strategy == 'balanced':
            return self._balanced_sampling(sequences, labels, metadata)
        
        elif strategy == 'target_focused':
            return self._target_focused_sampling(sequences, labels, metadata)
        
        elif strategy == 'progressive':
            return self._progressive_sampling(sequences, labels, metadata)
    
    def _balanced_sampling(self, sequences, labels, metadata, 
                          target_ratio=0.3, max_samples_per_class=2000):
        """平衡采样策略"""
        
        # 分类样本
        class_indices = {}
        for i, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        print("原始类别分布:")
        for label, indices in class_indices.items():
            print(f"  类别 {label}: {len(indices)} 个样本")
        
        # 计算采样数量
        n_target_samples = len(class_indices.get(1, [])) + len(class_indices.get(0, []))
        n_background_samples = min(
            int(n_target_samples / target_ratio * (1 - target_ratio)),
            len(class_indices.get(-1, [])),
            max_samples_per_class
        )
        
        # 采样
        selected_indices = []
        
        # 保留所有目标样本
        for label in [0, 1]:
            if label in class_indices:
                selected_indices.extend(class_indices[label])
        
        # 随机采样背景样本
        if -1 in class_indices:
            background_indices = np.random.choice(
                class_indices[-1], 
                size=min(n_background_samples, len(class_indices[-1])),
                replace=False
            )
            selected_indices.extend(background_indices)
        
        # 返回采样结果
        selected_indices = np.array(selected_indices)
        return (sequences[selected_indices], 
                labels[selected_indices], 
                [metadata[i] for i in selected_indices])
    
    def _progressive_sampling(self, sequences, labels, metadata):
        """渐进式采样 - 逐步增加负样本比例"""
        # 第一阶段：只用目标样本预训练
        target_indices = [i for i, label in enumerate(labels) if label != -1]
        
        # 第二阶段：添加少量负样本
        background_indices = [i for i, label in enumerate(labels) if label == -1]
        n_background = min(len(target_indices), len(background_indices))
        selected_background = np.random.choice(background_indices, n_background, replace=False)
        
        all_indices = target_indices + selected_background.tolist()
        
        return (sequences[all_indices], 
                labels[all_indices], 
                [metadata[i] for i in all_indices])

class SingleBeam1DCNN(nn.Module):
    """专为单波束时序数据设计的1D-CNN"""
    
    def __init__(self, sequence_length, num_classes=3, dropout_rate=0.4):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # 多尺度特征提取
        self.multi_scale_conv = nn.ModuleList([
            # 短期特征 (高频)
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ),
            # 中期特征 (中频)
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ),
            # 长期特征 (低频)
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            )
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_rate),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout1d(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length)
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv_block in self.multi_scale_conv:
            features = conv_block(x)
            multi_scale_features.append(features)
        
        # 特征拼接
        fused_features = torch.cat(multi_scale_features, dim=1)
        
        # 特征融合和池化
        features = self.feature_fusion(fused_features)
        features = features.view(features.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss处理类别不平衡"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_single_beam_model(sequences, labels, metadata, config=None):
    """训练单波束模型"""
    
    if config is None:
        config = {
            'sampling_strategy': 'balanced',
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 200,
            'n_splits': 5
        }
    
    # 数据处理
    processor = SingleBeamDataProcessor()
    
    # 智能采样
    X_sampled, y_sampled, meta_sampled = processor.intelligent_sampling(
        sequences, labels, metadata, strategy=config['sampling_strategy']
    )
    
    print(f"\n采样后数据分布:")
    unique, counts = np.unique(y_sampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  类别 {u}: {c} 个样本")
    
    # 标签映射 (-1,0,1) -> (0,1,2)
    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = np.array([label_map[y] for y in y_sampled])
    
    # 交叉验证训练
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sampled, y_mapped)):
        print(f"\n=== 第 {fold+1}/{config['n_splits']} 折 ===")
        
        X_train, X_val = X_sampled[train_idx], X_sampled[val_idx]
        y_train, y_val = y_mapped[train_idx], y_mapped[val_idx]
        
        # 创建模型
        model = SingleBeam1DCNN(sequence_length=X_train.shape[1])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        # 损失函数和优化器
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # 训练循环
        best_val_acc = 0
        patience = 30
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            # 训练阶段
            model.train()
            # ... 训练代码 ...
            
            # 验证阶段
            model.eval()
            # ... 验证代码 ...
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: 验证准确率待实现")
        
        fold_scores.append(0.85)  # 占位符
    
    print(f"\n交叉验证平均分数: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    return model, fold_scores

# 使用示例
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    multibeam_data = np.random.randn(60, 165, 100)  # 60 ping, 165 beam, 100 time samples
    target_info = [
        (10, 50, 1),   # ping 10, beam 50, target type 1
        (15, 75, 0),   # ping 15, beam 75, target type 0
        (25, 30, 1),   # ping 25, beam 30, target type 1
        # ... 更多目标信息
    ]
    
    # 处理数据
    processor = SingleBeamDataProcessor()
    sequences, labels, metadata = processor.extract_beam_sequences(multibeam_data, target_info)
    
    print(f"提取的序列数量: {len(sequences)}")
    print(f"序列维度: {sequences[0].shape}")
    print(f"标签分布: {Counter(labels)}")
    
    # 训练模型
    model, scores = train_single_beam_model(sequences, labels, metadata)
