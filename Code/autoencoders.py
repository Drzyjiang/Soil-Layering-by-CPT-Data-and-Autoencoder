import torch
import torch.nn as nn



class soilTransformer3(nn.Module):
    '''
    Purpose:
    An autoencoder with transformer layers
    positional encoding is learned, not prescribed
    Enable using mask to excluding padded positions
    
    '''
    def __init__(self, input_dim, embed_dim, seq_len, num_heads = 4, num_layers = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # learned positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        print(f"self.positional_encoding shape: {self.positional_encoding.shape}")
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_dim)
        )
   
    def getEmbeddings(self, x, src_key_padding_mask = None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        # Note: TransformerEncoder's forward arguments are {src, mask = None, src_key_padding_mask = None}
        encoded = self.encoder(x, mask = None, src_key_padding_mask = src_key_padding_mask)
        return encoded
    
    def forward(self, x, src_key_padding_mask = None):
        embeddings = self.getEmbeddings(x, src_key_padding_mask = src_key_padding_mask)
        return self.decoder(embeddings)
    
    def reconstructByLatent(self, embeddings):
        return self.decoder(embeddings)
    
    def getOriginalEmbeddings(self, x):
        return  self.embedding(x)

class soilTransformer4(nn.Module):
    '''
    Purpose:
    use zeros as positional encoding
    
    '''
    def __init__(self, input_dim, embed_dim, seq_len, num_heads = 4, num_layers = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # learned positional encoding
        #self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # mandate positional encoding as zero
        self.register_buffer("positional_encoding", torch.zeros(1, seq_len, embed_dim))
        

        print(f"self.positional_encoding shape: {self.positional_encoding.shape}")
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_dim)
        )
   
    def getEmbeddings(self, x, src_key_padding_mask = None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        # Note: TransformerEncoder's forward arguments are {src, mask = None, src_key_padding_mask = None}
        encoded = self.encoder(x, mask = None, src_key_padding_mask = src_key_padding_mask)
        return encoded
    
    def forward(self, x, src_key_padding_mask = None):
        embeddings = self.getEmbeddings(x, src_key_padding_mask = src_key_padding_mask)
        return self.decoder(embeddings)
    
    def reconstructByLatent(self, embeddings):
        return self.decoder(embeddings)
    
    def getOriginalEmbeddings(self, x):
        return  self.embedding(x)