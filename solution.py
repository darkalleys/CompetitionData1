import torch.nn as nn
import torch
import torch.nn.functional as F

class SVDNet(nn.Module):
    def __init__(self, dim=64, rank=32):
        super(SVDNet, self).__init__()
        self.dim = dim
        self.rank = rank
        
        # CNN branch for spatial feature extraction
        self.cnn_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Third conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # Adaptive pooling to 8x8
        )
        
        # Calculate CNN output size
        self.cnn_output_size = 512 * 8 * 8
        
        # RNN branch for temporal modeling
        # Treat spatial locations as sequence steps
        self.rnn_input_size = 2  # real and imaginary parts
        self.rnn_hidden_size = 256
        self.rnn_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Feature fusion
        self.fusion_dim = self.cnn_output_size + self.rnn_hidden_size * 2  # *2 for bidirectional
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Output heads for U, S, V
        self.head_U = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim * rank * 2)
        )
        
        self.head_S = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, rank)
        )
        
        self.head_V = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim * rank * 2)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x):  # x: [64, 64, 2]
        batch_size = 1  # Single sample processing
        
        # Ensure input is the right shape and add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension: [1, 64, 64, 2]
        
        # CNN branch: extract spatial features
        # Permute to [batch, channels, height, width] for conv2d
        x_cnn = x.permute(0, 3, 1, 2)  # [1, 2, 64, 64]
        cnn_features = self.cnn_layers(x_cnn)  # [1, 512, 8, 8]
        cnn_features = cnn_features.reshape(batch_size, -1)  # [1, 512*8*8]
        
        # RNN branch: treat spatial locations as sequence
        # Reshape input for RNN: [batch, seq_len, input_size]
        x_rnn = x.reshape(batch_size, -1, 2)  # [1, 64*64, 2]
        
        # Initialize hidden states
        h0 = torch.zeros(self.rnn_layers * 2, batch_size, self.rnn_hidden_size, device=x.device)
        c0 = torch.zeros(self.rnn_layers * 2, batch_size, self.rnn_hidden_size, device=x.device)
        
        # Forward through LSTM
        rnn_output, (hn, cn) = self.lstm(x_rnn, (h0, c0))  # [1, 64*64, 512]
        
        # Use the last hidden state as RNN features
        rnn_features = hn[-2:].transpose(0, 1).contiguous().reshape(batch_size, -1)  # [1, 512]
        
        # Feature fusion
        fused_features = torch.cat([cnn_features, rnn_features], dim=1)  # [1, fusion_dim]
        fused_features = self.fusion_layer(fused_features)  # [1, 512]
        
        # Generate outputs
        U_raw = self.head_U(fused_features).reshape(self.dim, self.rank, 2)  # [64, 32, 2]
        S_raw = self.head_S(fused_features).reshape(self.rank)  # [32]
        V_raw = self.head_V(fused_features).reshape(self.dim, self.rank, 2)  # [64, 32, 2]
        
        # Apply constraints
        U = self.enforce_orthogonality(U_raw)
        V = self.enforce_orthogonality(V_raw)
        S = self.enforce_singular_values(S_raw)
        
        return U, S, V
    
    def enforce_orthogonality(self, mat):
        """Enforce orthogonality constraint for complex matrices using QR decomposition"""
        # mat: [dim, rank, 2] where last dimension is [real, imag]
        real = mat[..., 0]  # [dim, rank]
        imag = mat[..., 1]  # [dim, rank]
        
        # Convert to complex tensor for QR decomposition
        complex_mat = torch.complex(real, imag)  # [dim, rank]
        
        # QR decomposition
        Q, R = torch.linalg.qr(complex_mat)  # Q: [dim, rank], R: [rank, rank]
        
        # Ensure proper phase (make R diagonal positive)
        R_diag = torch.diag(R)
        phase = R_diag / (torch.abs(R_diag) + 1e-8)
        Q = Q / phase.unsqueeze(0)
        
        # Convert back to real/imaginary representation
        Q_real = Q.real
        Q_imag = Q.imag
        
        return torch.stack([Q_real, Q_imag], dim=-1)
    
    def enforce_singular_values(self, s):
        """Enforce positive and descending order constraints for singular values"""
        # Apply ReLU for positivity
        s_positive = F.relu(s) + 1e-6  # Add small epsilon to avoid zeros
        
        # Sort in descending order
        s_sorted, _ = torch.sort(s_positive, descending=True)
        
        return s_sorted
    