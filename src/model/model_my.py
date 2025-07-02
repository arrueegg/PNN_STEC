import torch
import torch.nn as nn

def get_activation_fn(activation):
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim=1)  # Softmax requires specifying a dimension
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "swish":
        return nn.SiLU()  # SiLU is also known as Swish
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

# Initialization function
def init_xavier(model, activation, model_seed):
    def init_weights(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            gain = nn.init.calculate_gain(activation)
            torch.manual_seed(model_seed)  # Set the seed for reproducibility
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
            #torch.nn.init.xavier_uniform_(m.weight, gain=gain)  # Alternative (common choice for small nets)
            if m.bias is not None:
                m.bias.data.fill_(0)
    model.apply(init_weights)

def init_kaiming(model, activation, model_seed):
    def init_weights(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            nonlinearity = activation
            torch.manual_seed(model_seed)
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            if m.bias is not None:
                m.bias.data.fill_(0)
    model.apply(init_weights)
    #nn.init.constant_(model.out_layer.bias, 100.0)

class MLPBlock(nn.Module):
    """ 
    A modular MLP block consisting of:
    - Fully connected layer
    - Batch Normalization
    - Activation function
    - Dropout
    - Optional residual connection
    """
    def __init__(self, in_features, out_features, activation_fn, dropout_prob):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.LayerNorm(out_features)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_prob)
        self.use_residual = in_features == out_features

    def forward(self, x):
        residual = x if self.use_residual else None
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out) if self.training else out
        if residual is not None:
            out += residual
        return out

class MLPModel(nn.Module):
    """
    A gradient-stable MLP model with:
    - Batch Normalization
    - Dropout
    - Residual connections (when possible)
    - Optional Softplus output transformation (for positive-only outputs)
    """
    def __init__(self, config):
        """
        Args:
            input_dimension: Number of input features.
            output_dimension: Number of output features.
            hidden_sizes: List of hidden layer sizes.
            activation: Activation function to use ('relu', 'tanh', etc.).
            apply_softplus: If True, applies softplus to the final output.
            dropout_prob: Dropout probability for each hidden layer.
        """
        super(MLPModel, self).__init__()
        self.in_size = config['model']['input_size']
        self.out_size = config['model']['output_size']
        self.hidden_sizes = [10,10,10,10]
        self.activation = "ReLU"
        self.apply_softplus = config['model']['apply_softplus']
        self.dropout_prob = config['model']['dropout']

        self.activation_fn = get_activation_fn(self.activation)

        # Define MLP layers using MLPBlock
        layers = [MLPBlock(self.in_size, self.hidden_sizes[0], self.activation_fn, self.dropout_prob)]
        for i in range(1, len(self.hidden_sizes)):
            layers.append(MLPBlock(self.hidden_sizes[i-1], self.hidden_sizes[i], self.activation_fn, self.dropout_prob))
        
        self.blocks = nn.Sequential(*layers)
        
        # Final output layer (without BatchNorm or Activation)
        self.out_layer = nn.Linear(self.hidden_sizes[-1], self.out_size)

        if self.apply_softplus:
            self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.blocks(x)
        x = self.out_layer(x)
        if self.apply_softplus:
            x = self.softplus(x)
        return x

class ParallelMLP_simple(nn.Module):
    """
    A modified MLP that processes:
    - Station coordinates (e.g., lat, lon)
    - A configurable number of IPP coordinates (e.g., lat, lon or lat, lon, alt)
    - Shared global features (e.g., azimuth, elevation, time)
    
    The model outputs one prediction per IPP such that:
      • Each IPP's local features (from its coordinates) affect only its own output.
      • The station and shared features contribute to every output via a global branch.
      • If the number of IPP branches doesn't match the desired output size, an extra
        output neuron is produced to quantify uncertainty.
    """
    def __init__(self, config):
        super(ParallelMLP_simple, self).__init__()
        # Determine coordinate dimension:
        # If SH_encoding is used, include spherical harmonic terms; otherwise, just 2.
        self.coord_dim = (2 + config['preprocessing']['SH_degree'] ** 2) if config['preprocessing']['SH_encoding'] else 2 
        self.num_ipps = config['model']['num_layers']  # number of IPPs
        self.shared_input_dim = 6  # shared features dimension (e.g., azimuth, elevation, time)
        self.hidden_sizes = config['model']['hidden_size']
        self.activation = config['model']['activation']
        self.dropout_prob = config['model']['dropout']
        self.apply_softplus = config['model']['apply_softplus']
        self.output_size = config['model']['output_size']

        # Select activation function:
        self.activation_fn = get_activation_fn(self.activation)

        # Helper function: builds an MLP as a sequence of MLPBlock layers.
        def build_mlp(in_features, hidden_sizes):
            layers = []
            for hidden in hidden_sizes:
                layers.append(MLPBlock(in_features, hidden, self.activation_fn, self.dropout_prob))
                in_features = hidden
            return nn.Sequential(*layers)
        
        # --- Global Branch ---
        # Use the same multi-block MLP architecture for station and shared features:
        self.station_mlp = build_mlp(self.coord_dim, self.hidden_sizes)
        self.shared_mlp = build_mlp(self.shared_input_dim, self.hidden_sizes)
        # Combine station and shared outputs (each of dimension hidden_sizes[-1]) to produce a global feature:
        self.global_linear = nn.Linear(self.hidden_sizes[-1] * 2, self.hidden_sizes[-1])
        
        # --- Local (IPP) Branches ---
        # One IPP branch per IPP that only processes its own coordinates:
        self.ipp_mlps = nn.ModuleList([
            build_mlp(self.coord_dim, self.hidden_sizes) for _ in range(self.num_ipps)
        ])
        
        # --- Final Layers ---
        # Each final layer takes the concatenation of its IPP's local feature and the global feature.
        self.final_layers = nn.ModuleList([
            nn.Linear(self.hidden_sizes[-1] * 2, 1) for _ in range(self.num_ipps)
        ])

        # Add uncertainty output if number of IPP branches is not equal to desired output size.
        if self.num_ipps != self.output_size:
            self.uncertainty_layer = nn.Linear(self.hidden_sizes[-1], 1)
        else:
            self.uncertainty_layer = None

        if self.apply_softplus:
            self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Expected input tensor shape:
         - Station coordinates: first `coord_dim` values.
         - IPP coordinates: next `num_ipps * coord_dim` values.
         - Shared features: remaining values (should equal shared_input_dim).
        """

        # Extract batch size:
        batch_size = x.shape[0]

        # Extract station coordinates:
        station_coords = x[:, :self.coord_dim]
        
        # Extract IPP coordinates dynamically:
        ipp_end = self.coord_dim + self.num_ipps * self.coord_dim
        ipp_coords = x[:, self.coord_dim:ipp_end].view(batch_size, self.num_ipps, self.coord_dim)
        
        # Extract shared features:
        shared_features = x[:, ipp_end:]
        
        # Process the global features:
        station_out = self.station_mlp(station_coords)  # shape: [batch, hidden_sizes[-1]]
        shared_out = self.shared_mlp(shared_features)    # shape: [batch, hidden_sizes[-1]]
        global_cat = torch.cat([station_out, shared_out], dim=-1)  # shape: [batch, 2 * hidden_sizes[-1]]
        global_feature = self.activation_fn(self.global_linear(global_cat))  # shape: [batch, hidden_sizes[-1]]
        
        # Process each IPP's local features:
        ipp_outs = [self.ipp_mlps[i](ipp_coords[:, i, :]) for i in range(self.num_ipps)]
        
        # For each IPP, concatenate its local feature with the global feature and predict:
        final_outputs = [
            self.final_layers[i](torch.cat([ipp_outs[i], global_feature], dim=-1)).squeeze(-1)
            for i in range(self.num_ipps)
        ]
        predictions = torch.stack(final_outputs, dim=1)  # shape: [batch, num_ipps]

        # If uncertainty output is required, compute it from the global feature and append it.
        if self.uncertainty_layer is not None:
            uncertainty = self.uncertainty_layer(global_feature).squeeze(-1)  # shape: [batch]
            # Append uncertainty as an extra column to the predictions.
            output = torch.cat([predictions, uncertainty.unsqueeze(1)], dim=1)  # shape: [batch, num_ipps+1]
        else:
            output = predictions

        if self.apply_softplus:
            output = self.softplus(output)
        return output

class ParallelMLP(nn.Module):
    """
    A modified MLP that processes:
    - Station coordinates (e.g., lat, lon)
    - A configurable number of IPP coordinates (e.g., lat, lon or lat, lon, alt)
    - Shared global features (e.g., azimuth, elevation, time)
    
    The model outputs one prediction per IPP such that:
      • Each IPP's local features (from its coordinates) affect only its own output.
      • The station and shared features are merged and processed by a global branch,
        allowing them to learn from each other and contribute to every output.
      • The IPP branches are designed as smaller networks since they only provide the final correction.
      • If the number of IPP branches doesn't match the desired output size, an extra
        output neuron is produced to quantify uncertainty.
    """
    def __init__(self, config):
        super(ParallelMLP, self).__init__()
        # Determine coordinate dimension:
        self.coord_dim = (2 + config['preprocessing']['SH_degree'] ** 2) if config['preprocessing']['SH_encoding'] else 2 
        self.num_ipps = config['model']['num_layers']  # number of IPPs
        self.shared_input_dim = 6  # shared features dimension (e.g., azimuth, elevation, time)
        self.hidden_sizes = config['model']['hidden_size']
        self.activation = config['model']['activation']
        self.dropout_prob = config['model']['dropout']
        self.apply_softplus = config['model']['apply_softplus']
        self.output_size = config['model']['output_size']
        
        # Global branch input: station + shared features.
        self.global_input_dim = self.coord_dim + self.shared_input_dim

        # Select activation function:
        self.activation_fn = get_activation_fn(self.activation)
        
        # Helper function: builds an MLP as a sequence of MLPBlock layers.
        def build_mlp(in_features, hidden_sizes):
            layers = []
            for hidden in hidden_sizes:
                layers.append(MLPBlock(in_features, hidden, self.activation_fn, self.dropout_prob))
                in_features = hidden
            return nn.Sequential(*layers)
        
        # --- Global Branch ---
        # Merge station and shared features:
        self.global_mlp = build_mlp(self.global_input_dim, self.hidden_sizes)
        
        # --- Local (IPP) Branches ---
        # Use a smaller network for each IPP branch since these inputs are less complex.
        # Optionally, provide a custom list via config['model'].get('ipp_hidden_size', ...)
        ipp_hidden_sizes = config['model'].get('ipp_hidden_size', [self.hidden_sizes[0]])
        # Each IPP branch: a smaller MLP followed by a linear mapping to global feature dimension.
        self.ipp_mlps = nn.ModuleList([
            nn.Sequential(
                build_mlp(self.coord_dim, ipp_hidden_sizes),
                nn.Linear(ipp_hidden_sizes[-1], self.hidden_sizes[-1]),
                self.activation_fn
            ) for _ in range(self.num_ipps)
        ])
        
        # --- Final Layers ---
        # Each final layer takes the concatenation of its IPP's local feature and the global feature.
        self.final_layers = nn.ModuleList([
            nn.Linear(self.hidden_sizes[-1] * 2, 1) for _ in range(self.num_ipps)
        ])
        
        # Add uncertainty output if number of IPP branches is not equal to desired output size.
        if self.num_ipps != self.output_size:
            self.uncertainty_layer = nn.Linear(self.hidden_sizes[-1], 1)
        else:
            self.uncertainty_layer = None
        
        if self.apply_softplus:
            self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Expected input tensor shape:
         - Station coordinates: first `coord_dim` values.
         - IPP coordinates: next `num_ipps * coord_dim` values.
         - Shared features: remaining values (should equal shared_input_dim).
        """
        batch_size = x.shape[0]
        # Extract station coordinates:
        station_coords = x[:, :self.coord_dim]
        
        # Extract IPP coordinates dynamically:
        ipp_end = self.coord_dim + self.num_ipps * self.coord_dim
        ipp_coords = x[:, self.coord_dim:ipp_end].view(batch_size, self.num_ipps, self.coord_dim)
        
        # Extract shared features:
        shared_features = x[:, ipp_end:]
        
        # --- Global Branch ---
        # Concatenate station and shared features and process them together:
        global_input = torch.cat([station_coords, shared_features], dim=-1)
        global_feature = self.global_mlp(global_input)  # shape: [batch, hidden_sizes[-1]]
        
        # --- Local (IPP) Branches ---
        # Process each IPP's local features:
        ipp_outs = [self.ipp_mlps[i](ipp_coords[:, i, :]) for i in range(self.num_ipps)]
        
        # For each IPP, concatenate its local feature with the global feature and predict:
        final_outputs = [
            self.final_layers[i](torch.cat([ipp_outs[i], global_feature], dim=-1)).squeeze(-1)
            for i in range(self.num_ipps)
        ]
        predictions = torch.stack(final_outputs, dim=1)  # shape: [batch, num_ipps]
        
        # If uncertainty output is required, compute it from the global feature and append it.
        if self.uncertainty_layer is not None:
            uncertainty = self.uncertainty_layer(global_feature).squeeze(-1)  # shape: [batch]
            output = torch.cat([predictions, uncertainty.unsqueeze(1)], dim=1)  # shape: [batch, num_ipps+1]
        else:
            output = predictions
        
        if self.apply_softplus:
            output = self.softplus(output)
        return output

class ParallelTransformer(nn.Module):
    """
    A transformer-based model that processes:
      - Station coordinates (e.g., lat, lon)
      - A configurable number of IPP coordinates (e.g., lat, lon or lat, lon, alt)
      - Shared global features (e.g., azimuth, elevation, time)
    
    Input format (per sample):
      • First `coord_dim` values: station coordinates.
      • Next `num_ipps * coord_dim` values: IPP coordinates.
      • Remaining values: shared features.
    
    The model outputs one prediction per IPP such that:
      • The global token (from station and shared features) provides context for all IPPs.
      • Each IPP token (projected from its coordinates) provides a local correction.
      • If the number of IPP tokens doesn't match the desired output size, an extra output neuron
        is produced to quantify uncertainty.
    """
    def __init__(self, config):
        super(ParallelTransformer, self).__init__()
        # Determine coordinate dimension:
        # If SH_encoding is used, include spherical harmonic terms; otherwise, just 2.
        self.coord_dim = (2 + config['preprocessing']['SH_degree'] ** 2) if config['preprocessing']['SH_encoding'] else 2
        self.num_ipps = config['model']['num_layers']  # number of IPPs
        self.shared_input_dim = 6  # e.g., azimuth, elevation, time
        self.output_size = config['model']['output_size']
        self.dropout_prob = config['model']['dropout']
        self.apply_softplus = config['model']['apply_softplus']

        # We'll use the final hidden size from the MLP config as the transformer model dimension.
        # (You could also set this separately in the config.)
        self.d_model = config['model']['hidden_size'][-1]
        
        # Global branch projection: combine station and shared features.
        global_in_dim = self.coord_dim + self.shared_input_dim
        self.global_proj = nn.Linear(global_in_dim, self.d_model)
        # IPP branch projection: project each IPP coordinate into the model dimension.
        self.ipp_proj = nn.Linear(self.coord_dim, self.d_model)
        
        # Create a learnable positional embedding for each token.
        # We have 1 global token + num_ipps IPP tokens.
        self.pos_embedding = nn.Parameter(torch.randn(self.num_ipps + 1, self.d_model))
        
        # Transformer Encoder:
        # You can set the number of transformer layers and heads via config.
        num_transformer_layers = config['model'].get('num_transformer_layers', 2)
        num_heads = config['model'].get('num_heads', 4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dropout=self.dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Final prediction layers: for each IPP token, we combine its encoded representation with the global token.
        self.final_layers = nn.ModuleList([
            nn.Linear(self.d_model * 2, 1) for _ in range(self.num_ipps)
        ])
        
        # Uncertainty output if desired:
        if self.num_ipps != self.output_size:
            self.uncertainty_layer = nn.Linear(self.d_model, 1)
        else:
            self.uncertainty_layer = None
        
        if self.apply_softplus:
            self.softplus = nn.Softplus()
    
    def forward(self, x):
        """
        Expected input tensor shape:
          - Station coordinates: first `coord_dim` values.
          - IPP coordinates: next `num_ipps * coord_dim` values.
          - Shared features: remaining values (should equal shared_input_dim).
        """
        batch_size = x.shape[0]
        # Extract station coordinates:
        station_coords = x[:, :self.coord_dim]  # shape: (batch, coord_dim)
        # Extract IPP coordinates:
        ipp_end = self.coord_dim + self.num_ipps * self.coord_dim
        ipp_coords = x[:, self.coord_dim:ipp_end].view(batch_size, self.num_ipps, self.coord_dim)  # shape: (batch, num_ipps, coord_dim)
        # Extract shared features:
        shared_features = x[:, ipp_end:]  # shape: (batch, shared_input_dim)
        
        # Global token: concatenate station and shared features and project.
        global_input = torch.cat([station_coords, shared_features], dim=-1)  # shape: (batch, coord_dim+shared_input_dim)
        global_token = self.global_proj(global_input)  # shape: (batch, d_model)
        
        # IPP tokens: project each IPP coordinate.
        ipp_tokens = self.ipp_proj(ipp_coords)  # shape: (batch, num_ipps, d_model)
        
        # Form a sequence: prepend the global token to the IPP tokens.
        tokens = torch.cat([global_token.unsqueeze(1), ipp_tokens], dim=1)  # shape: (batch, num_ipps+1, d_model)
        # Add positional embeddings.
        tokens = tokens + self.pos_embedding.unsqueeze(0)  # shape: (batch, num_ipps+1, d_model)
        
        # Transformer expects input shape (sequence_length, batch_size, d_model)
        tokens = tokens.transpose(0, 1)  # shape: (num_ipps+1, batch, d_model)
        encoded_tokens = self.transformer_encoder(tokens)  # shape: (num_ipps+1, batch, d_model)
        encoded_tokens = encoded_tokens.transpose(0, 1)  # shape: (batch, num_ipps+1, d_model)
        
        # Separate the global token (first token) and the IPP tokens.
        global_encoded = encoded_tokens[:, 0, :]  # shape: (batch, d_model)
        ipp_encoded = encoded_tokens[:, 1:, :]      # shape: (batch, num_ipps, d_model)
        
        # For each IPP, concatenate its token with the global token and predict.
        final_outputs = [
            self.final_layers[i](torch.cat([ipp_encoded[:, i, :], global_encoded], dim=-1)).squeeze(-1)
            for i in range(self.num_ipps)
        ]
        predictions = torch.stack(final_outputs, dim=1)  # shape: (batch, num_ipps)
        
        # Uncertainty output from the global token if needed.
        if self.uncertainty_layer is not None:
            uncertainty = self.uncertainty_layer(global_encoded).squeeze(-1)  # shape: (batch)
            output = torch.cat([predictions, uncertainty.unsqueeze(1)], dim=1)  # shape: (batch, num_ipps+1)
        else:
            output = predictions
        
        if self.apply_softplus:
            output = self.softplus(output)
        return output

class UnifiedTransformer(nn.Module):
    """
    A transformer-based model that processes all inputs as a single unified sequence.
    
    Input format (per sample):
      - First `coord_dim` values: station coordinates.
      - Next `num_ipps * coord_dim` values: IPP coordinates.
      - Remaining values: shared global features.
    
    The model creates three token types:
      • A station token from the station coordinates.
      • One token per IPP coordinate.
      • A shared token from the shared features.
      
    These tokens (after embedding and adding positional encodings) are processed
    jointly by a transformer encoder. The output representations for the IPP tokens
    are then used to produce one prediction per IPP, with an extra uncertainty output
    if required.
    """
    def __init__(self, config):
        super(UnifiedTransformer, self).__init__()
        # Determine coordinate dimension:
        # If SH_encoding is used, include spherical harmonic terms; otherwise, just 2.
        self.coord_dim = (2 + config['preprocessing']['SH_degree'] ** 2) if config['preprocessing']['SH_encoding'] else 2
        self.num_ipps = config['model']['num_layers']  # number of IPPs
        self.shared_input_dim = 6  # e.g., azimuth, elevation, time
        self.output_size = config['model']['output_size']
        self.dropout_prob = config['model']['dropout']
        self.apply_softplus = config['model']['apply_softplus']
        
        # Transformer model dimension (you can also set this separately)
        self.d_model = config['model']['model_dim']
        
        # --- Embedding layers for each token type ---
        self.station_embed = nn.Linear(self.coord_dim, self.d_model)
        self.ipp_embed = nn.Linear(self.coord_dim, self.d_model)
        self.shared_embed = nn.Linear(self.shared_input_dim, self.d_model)
        
        # Total sequence length: station (1) + IPPs (num_ipps) + shared (1)
        self.seq_length = self.num_ipps + 2
        self.pos_embedding = nn.Parameter(torch.randn(self.seq_length, self.d_model))
        
        # --- Transformer Encoder ---
        num_transformer_layers = config['model'].get('num_transformer_layers', 2)
        num_heads = config['model'].get('num_heads', 4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dropout=self.dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # --- Final Prediction Heads ---
        # Each IPP token (positions 1 to num_ipps) is mapped to a prediction.
        self.final_heads = nn.ModuleList([
            nn.Linear(self.d_model, 1) for _ in range(self.num_ipps)
        ])
        # Additional uncertainty output if needed:
        if self.num_ipps != self.output_size:
            # We use the shared token representation for uncertainty.
            self.uncertainty_head = nn.Linear(self.d_model, 1)
        else:
            self.uncertainty_head = None
        
        if self.apply_softplus:
            self.softplus = nn.Softplus()
    
    def forward(self, x):
        """
        Expected input tensor shape:
          - Station coordinates: first `coord_dim` values.
          - IPP coordinates: next `num_ipps * coord_dim` values.
          - Shared features: remaining values (should equal shared_input_dim).
        """
        batch_size = x.shape[0]
        # --- Token Extraction ---
        # Station token:
        station = x[:, :self.coord_dim]  # shape: (batch, coord_dim)
        # IPP tokens:
        ipp_start = self.coord_dim
        ipp_end = self.coord_dim + self.num_ipps * self.coord_dim
        ipp = x[:, ipp_start:ipp_end].view(batch_size, self.num_ipps, self.coord_dim)  # shape: (batch, num_ipps, coord_dim)
        # Shared token:
        shared = x[:, ipp_end:]  # shape: (batch, shared_input_dim)
        
        # --- Embedding ---
        station_token = self.station_embed(station)        # (batch, d_model)
        ipp_tokens = self.ipp_embed(ipp)                     # (batch, num_ipps, d_model)
        shared_token = self.shared_embed(shared)             # (batch, d_model)
        
        # Build sequence: [station_token, ipp_tokens..., shared_token]
        station_token = station_token.unsqueeze(1)  # (batch, 1, d_model)
        shared_token = shared_token.unsqueeze(1)      # (batch, 1, d_model)
        tokens = torch.cat([station_token, ipp_tokens, shared_token], dim=1)  # (batch, seq_length, d_model)
        
        # Add positional embeddings:
        tokens = tokens + self.pos_embedding.unsqueeze(0)  # (batch, seq_length, d_model)
        
        # --- Transformer Encoding ---
        # Transformer expects (sequence_length, batch, d_model)
        tokens = tokens.transpose(0, 1)  # (seq_length, batch, d_model)
        encoded_tokens = self.transformer_encoder(tokens)  # (seq_length, batch, d_model)
        encoded_tokens = encoded_tokens.transpose(0, 1)  # (batch, seq_length, d_model)
        
        # --- Final Prediction ---
        # Use the IPP tokens (positions 1 to num_ipps) for predictions.
        ipp_encoded = encoded_tokens[:, 1:1+self.num_ipps, :]  # (batch, num_ipps, d_model)
        predictions = []
        for i in range(self.num_ipps):
            pred = self.final_heads[i](ipp_encoded[:, i, :]).squeeze(-1)  # (batch,)
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # (batch, num_ipps)
        
        # Uncertainty output (if required) is computed from the shared token (last token).
        if self.uncertainty_head is not None:
            shared_encoded = encoded_tokens[:, -1, :]  # (batch, d_model)
            uncertainty = self.uncertainty_head(shared_encoded).squeeze(-1)  # (batch,)
            output = torch.cat([predictions, uncertainty.unsqueeze(1)], dim=1)  # (batch, num_ipps+1)
        else:
            output = predictions
        
        if self.apply_softplus:
            output = self.softplus(output)
        return output


# Model selection function
def get_model(config):
    model_type = config['model']['model_type']
    if model_type == 'MLP':
        return MLPModel(config)
    elif model_type == 'ParallelMLP':
        return ParallelMLP_simple(config)
    elif model_type == 'Transformer':
        return UnifiedTransformer(config)
    else:
        raise ValueError(f"Model type {model_type} is not recognized. Please select from ['MLP', 'RNN']")

