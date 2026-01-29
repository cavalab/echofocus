"""Model architectures for EchoFocus."""

import torch
from torch import nn

class CustomDropout(nn.Module):
    """Custom dropout that drops entire clip embeddings."""

    def __init__(self, p):
        """Initialize the dropout module.

        Args:
            p (float): Dropout probability for entire clip embeddings.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """Apply dropout by dropping entire clip rows during training.

        Args:
            x (torch.Tensor): Input tensor shaped (B, T, D).

        Returns:
            torch.Tensor: Tensor with some rows removed in training mode.
        """
        if self.training:  # apply dropout only during training
            row_keep_mask = (
                torch.rand(x.shape[1]) > self.p
            )  # p = 0.8 -> 80% dropout -> keep 20%
            x = x[:, row_keep_mask, :]  # py_torch can propagate through this

            return x
        else:
            return x
class CustomTransformer(nn.Module):
    """Transformer encoder over clip embeddings with pooling."""

    # if n_layers = 0 -> MHA
    def __init__ (self, input_size=768, encoder_dim=768, n_encoder_layers=0, output_size=1, clip_dropout = 0, tf_combine = 'avg'):
        """Initialize the transformer model for clip embeddings.

        Args:
            input_size (int): Input embedding dimension.
            encoder_dim (int): Feed-forward dimension in the encoder.
            n_encoder_layers (int): Number of transformer encoder layers.
            output_size (int): Number of output targets.
            clip_dropout (float): Dropout probability for clip embeddings.
            tf_combine (str): Pooling method: ``"avg"`` or ``"max"``.
        """
        super(CustomTransformer, self).__init__()
        
        self.clip_dropout = CustomDropout(clip_dropout)
        self.tf_combine = tf_combine
        
        N_heads = 6
        if (n_encoder_layers ==0):
            self.encoder = nn.MultiheadAttention(input_size, N_heads,0.2,batch_first=True)
        else:
            enc_layer = nn.TransformerEncoderLayer(input_size, N_heads, encoder_dim, 0.2, batch_first=True) # batch_first has no effect (feeding one sequence at a time), but hides a warning 
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_encoder_layers)

        # final linear
        self.ff = nn.Linear(in_features = encoder_dim, out_features = output_size) 

    def embed(self, x):
        """Return pooled encoder representation for a set of clips.

        Args:
            x (iterable[torch.Tensor]): Sequence of clip embeddings.

        Returns:
            torch.Tensor: Pooled representation vector.
        """
        x = torch.vstack([k for k in x]) # combine the tensors for all the videos
        x = x.unsqueeze(0)
        # y = x.squeeze()
        
        x = self.clip_dropout(x) # drop some video embeddings entirely

        # pretend this is a sequence, and we only care about the encoding after having seen the whole thing
        # out = self.encoder(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]), is_causal  = True)
        # out = out[:,-1,:]
        
        out = self.encoder(x)
        if (self.tf_combine == 'avg'):
            out = torch.mean(out[0],axis=0) # average the representation?
        elif (self.tf_combine == 'max'):            
            out,__ = torch.max(out[0],dim=0)

        return out

    def forward(self, x):
        """Compute model outputs for a set of clips.

        Args:
            x (iterable[torch.Tensor]): Sequence of clip embeddings.

        Returns:
            torch.Tensor: Output logits or regression values.
        """
        x = torch.vstack([k for k in x]) # combine the tensors for all the videos
        x = x.unsqueeze(0)
        # y = x.squeeze()
        
        x = self.clip_dropout(x) # drop some video embeddings entirely

        # pretend this is a sequence, and we only care about the encoding after having seen the whole thing
        # out = self.encoder(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]), is_causal  = True)
        # out = out[:,-1,:]
        
        out = self.encoder(x)
        
        if (self.tf_combine == 'avg'):
            out = torch.mean(out[0],axis=0) # average the representation?
        elif (self.tf_combine == 'max'):            
            out,__ = torch.max(out[0],dim=0)
        linear_out = self.ff(out)

        return linear_out
        
class CustomQueryTransformer(nn.Module):
    """Transformer with a learned query token for set pooling."""

    def __init__(
        self,
        input_size=768,
        encoder_dim=768,        # kept for API-compat; see note below
        n_encoder_layers=0,
        output_size=1,
        clip_dropout=0,
        n_heads=6,
    ):
        """Initialize a query-token transformer for set pooling.

        Args:
            input_size (int): Input embedding dimension.
            encoder_dim (int): Feed-forward dimension in the encoder layer.
            n_encoder_layers (int): Number of encoder layers.
            output_size (int): Number of output targets.
            clip_dropout (float): Dropout probability for clip embeddings.
            n_heads (int): Number of attention heads.
        """
        super().__init__()

        self.clip_dropout = CustomDropout(clip_dropout)

        # Learned query/[CLS] token that will pool information across the set of videos
        # Shape: (1, 1, d) so it can be concatenated along sequence dimension
        self.query_token = nn.Parameter(torch.zeros(1, 1, input_size))
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

        if n_encoder_layers == 0:
            # Multi-head self-attention block (batch_first=True => (B, T, D))
            self.encoder = nn.MultiheadAttention(
                embed_dim=input_size,
                num_heads=n_heads,
                dropout=0.2,
                batch_first=True,
            )
            ff_in_dim = input_size
        else:
            # NOTE: In PyTorch, dim_feedforward is the MLP hidden size inside the layer.
            # Your old code used encoder_dim there; we preserve that behavior.
            enc_layer = nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=n_heads,
                dim_feedforward=encoder_dim,
                dropout=0.2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
            ff_in_dim = input_size

        # Final linear head (multitask logits)
        self.ff = nn.Linear(in_features=ff_in_dim, out_features=output_size)

    def forward(self, x):
        """Compute model outputs for a set of video embeddings.

        Args:
            x (iterable[torch.Tensor]): Video embeddings shaped (D,).

        Returns:
            torch.Tensor: Output logits or regression values.
        """
        # (n_videos, D)
        # x = torch.stack(list(x), dim=0)
        x = torch.vstack(list(x)) 

        # (1, n_videos, D)
        x = x.unsqueeze(0)

        # Drop some video embeddings entirely DURING TRAINING ONLY (your CustomDropout)
        x = self.clip_dropout(x)  # still shape (1, n_kept_videos, D) in training; unchanged in eval

        # Prepend query token AFTER dropout so it is never dropped
        q = self.query_token.to(x.dtype).to(x.device).expand(x.shape[0], -1, -1)  # (1,1,D)
        x = torch.cat([q, x], dim=1)  # (1, 1 + n_videos, D)

        # Encoder
        if isinstance(self.encoder, nn.MultiheadAttention):
            # MultiheadAttention returns (attn_output, attn_weights)
            attn_output, _ = self.encoder(x, x, x, need_weights=False)
            out = attn_output
        else:
            out = self.encoder(x)

        # Pool: take the query token output
        pooled = out[:, 0, :]          # (1, D)

        # Head: multitask logits
        linear_out = self.ff(pooled)   # (1, output_size)

        return linear_out.squeeze(0)   # (output_size,)

# class CustomTransformer (nn.Module):
#     # if n_layers = 0 -> MHA
#     def __init__ (self, input_size=768, encoder_dim=768, n_encoder_layers=0, output_size=1, clip_dropout = 0, tf_combine = 'avg'):
#         super(Custom_Transformer, self).__init__()
        
#         self.clip_dropout = CustomDropout(clip_dropout)
#         self.tf_combine = tf_combine
        
#         N_heads = 6
#         if (n_encoder_layers ==0):
#             self.encoder = nn.MultiheadAttention(input_size, N_heads,0.2,batch_first=True)
#         else:
#             enc_layer = nn.TransformerEncoderLayer(input_size, N_heads, encoder_dim, 0.2, batch_first=True) # batch_first has no effect (feeding one sequence at a time), but hides a warning 
#             self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_encoder_layers)

#         # final linear
#         self.ff = nn.Linear(in_features = encoder_dim, out_features = output_size) 
        
#     def forward(self, x):
        
#         x = torch.vstack([k for k in x]) # combine the tensors for all the videos
#         x = x.unsqueeze(0)
#         # y = x.squeeze()
        
#         x = self.clip_dropout(x) # drop some video embeddings entirely

#         # pretend this is a sequence, and we only care about the encoding after having seen the whole thing
#         # out = self.encoder(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]), is_causal  = True)
#         # out = out[:,-1,:]
#         if n_encoder_layers == 0:        
#             out = self.encoder(x,x,x)
#         else:
#             out = self.encoder(x)
        
#         if (self.tf_combine == 'avg'):
#             out = torch.mean(out[0],axis=0) # average the representation?
#         elif (self.tf_combine == 'max'):            
#             out,__ = torch.max(out[0],dim=0)
#         linear_out = self.ff(out)

#         return linear_out
