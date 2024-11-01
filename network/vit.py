from keras.src import layers, Model
import tensorflow as tf


class PatchEmbed(layers.Layer):
    """
    Patch embedding layer for transforming images into patch-based embeddings.

    @Usage:
        This layer divides an input image into non-overlapping patches, then applies
        a convolution to convert each patch into a dense vector (embedding).

    @Parameters:
    img_size : int
        Size of the input image (assumed square).
    patch_size : int
        Size of each patch (assumed square).
    embed_dim : int, optional, default=64
        Dimension of the embedding for each patch.

    @Attributes:
    n_patches : int
        Total number of patches created from the input image.
    proj : tf.keras.layers.Conv2D
        Convolutional layer that generates embeddings for each patch.
    """

    def __init__(self, img_size, patch_size, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def call(self, x):
        """
        Forward pass to generate patch embeddings.

        @Parameters:
        x : tf.Tensor
            Input tensor with shape (batch_size, img_size, img_size, channels).

        @Returns:
        tf.Tensor
            Output tensor with shape (batch_size, n_patches, embed_dim), where each
            patch is embedded as a vector of dimension `embed_dim`.
        """
        x = self.proj(x)
        x = tf.reshape(
            x, [tf.shape(x)[0], -1, tf.shape(x)[-1]]
        )  # (n_samples, n_patches, embed_dim)
        return x


class Attention(layers.Layer):
    """
    Multi-head self-attention layer.

    @Usage:
        This layer performs attention on the input sequence, using multiple attention
        heads to learn different aspects of the input features.

    @Parameters:
    dim : int
        Dimension of input and output features.
    n_heads : int, optional, default=4
        Number of attention heads.
    qkv_bias : bool, optional, default=True
        If True, includes bias in the query, key, and value projections.
    attn_p : float, optional, default=0.0
        Dropout rate for attention weights.
    proj_p : float, optional, default=0.0
        Dropout rate for the output projection.

    @Attributes:
    qkv : tf.keras.layers.Dense
        Linear projection layer for query, key, and value matrices.
    attn_drop : tf.keras.layers.Dropout
        Dropout layer applied to attention weights.
    proj : tf.keras.layers.Dense
        Linear projection layer for the output.
    proj_drop : tf.keras.layers.Dropout
        Dropout layer applied after the final projection.
    """

    def __init__(self, dim, n_heads=4, qkv_bias=True, attn_p=0.0, proj_p=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_p)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_p)

    def call(self, x):
        """
        Forward pass for multi-head attention.

        @Parameters:
        x : tf.Tensor
            Input tensor with shape (batch_size, n_tokens, dim).

        @Returns:
        tf.Tensor
            Output tensor with shape (batch_size, n_tokens, dim), where each token
            has attended information from other tokens.
        """
        n_samples = tf.shape(x)[0]  # Batch size
        n_tokens = tf.shape(x)[1]  # Number of tokens

        # Project input to Q, K, V and reshape
        qkv = self.qkv(x)  # Shape: (n_samples, n_tokens, 3 * dim)
        qkv = tf.reshape(qkv, [n_samples, n_tokens, 3, self.n_heads, self.head_dim])
        qkv = tf.transpose(
            qkv, perm=[2, 0, 3, 1, 4]
        )  # (3, n_samples, n_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into q, k, v

        # Compute scaled dot-product attention
        dp = (
            tf.matmul(q, k, transpose_b=True) * self.scale
        )  # (n_samples, n_heads, n_tokens, n_tokens)
        attn = tf.nn.softmax(dp, axis=-1)
        attn = self.attn_drop(attn)  # Apply dropout to attention scores

        # Weighted average of values
        weighted_avg = tf.matmul(attn, v)  # (n_samples, n_heads, n_tokens, head_dim)
        weighted_avg = tf.transpose(
            weighted_avg, perm=[0, 2, 1, 3]
        )  # (n_samples, n_tokens, n_heads, head_dim)

        # Reshape to (n_samples, n_tokens, dim)
        weighted_avg = tf.reshape(
            weighted_avg, [n_samples, n_tokens, self.n_heads * self.head_dim]
        )

        # Final linear projection
        x = self.proj(weighted_avg)  # (n_samples, n_tokens, dim)
        x = self.proj_drop(x)  # Apply dropout after projection
        return x


class MLP(layers.Layer):
    """
    Multi-layer perceptron (MLP) block.

    @Usage:
        This layer is used in each transformer block to apply a feedforward neural network.

    @Parameters:
    hidden_features : int
        Number of hidden units in the first layer.
    out_features : int
        Number of output units, typically the same as input dimension.
    p : float, optional, default=0.0
        Dropout rate.

    @Attributes:
    fc1 : tf.keras.layers.Dense
        First fully connected layer.
    fc2 : tf.keras.layers.Dense
        Second fully connected layer.
    drop : tf.keras.layers.Dropout
        Dropout layer applied after each dense layer.
    """

    def __init__(self, hidden_features, out_features, p=0.0, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden_features, activation="gelu")
        self.drop = layers.Dropout(p)
        self.fc2 = layers.Dense(out_features)

    def call(self, x):
        """
        Forward pass for the MLP.

        @Parameters:
        x : tf.Tensor
            Input tensor with shape (batch_size, n_tokens, in_features).

        @Returns:
        tf.Tensor
            Output tensor with shape (batch_size, n_tokens, out_features).
        """
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(layers.Layer):
    """
    Transformer block consisting of self-attention and MLP layers.

    @Usage:
        This layer applies attention to the input, followed by an MLP layer, with residual
        connections and layer normalization.

    @Parameters:
    dim : int
        Dimension of input and output.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        Ratio of hidden layer size to input size in the MLP.
    qkv_bias : bool, optional, default=True
        Whether to include bias in query, key, and value projections.
    p : float, optional, default=0.0
        Dropout rate for MLP.
    attn_p : float, optional, default=0.0
        Dropout rate for attention.

    @Attributes:
    norm1 : tf.keras.layers.LayerNormalization
        Layer normalization applied before attention.
    attn : Attention
        Multi-head attention layer.
    norm2 : tf.keras.layers.LayerNormalization
        Layer normalization applied before MLP.
    mlp : MLP
        Feedforward neural network layer.
    """

    def __init__(
        self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(hidden_features=hidden_features, out_features=dim, p=p)

    def call(self, x):
        """
        Forward pass for the transformer block.

        @Parameters:
        x : tf.Tensor
            Input tensor with shape (batch_size, n_tokens, dim).

        @Returns:
        tf.Tensor
            Output tensor with the same shape as the input (batch_size, n_tokens, dim).
        """
        x_norm1 = self.norm1(x)
        attn_out = self.attn(x_norm1)
        x = x + attn_out

        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        return x


class VisionTransformer(Model):
    """
    Vision Transformer (ViT) model for image classification.

    @Usage:
        This model uses transformer blocks for image classification by representing
        images as a sequence of patch embeddings.

    @Parameters:
    img_size : int, optional, default=28
        Height and width of the input image.
    patch_size : int, optional, default=7
        Height and width of each image patch.
    embed_dim : int, optional, default=64
        Dimensionality of the patch embeddings.
    depth : int, optional, default=6
        Number of transformer blocks.
    n_heads : int, optional, default=4
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        Ratio of hidden layer size to input size in the MLP.
    qkv_bias : bool, optional, default=True
        If True, include bias in query, key, and value projections.
    p : float, optional, default=0.0
        Dropout rate for MLP.
    attn_p : float, optional, default=0.0
        Dropout rate for attention.
    n_classes : int, optional, default=10
        Number of output classes.

    @Attributes:
    patch_embed : PatchEmbed
        Layer that converts image into patch embeddings.
    cls_token : tf.Variable
        Token representing the image class.
    pos_embed : tf.Variable
        Positional embeddings for patches and class token.
    pos_drop : tf.keras.layers.Dropout
        Dropout layer applied to embeddings.
    blocks : list of Block
        List of transformer blocks.
    norm : tf.keras.layers.LayerNormalization
        Layer normalization applied to transformer output.
    head : tf.keras.layers.Dense
        Dense layer for output classification.
    abbreviation : str
        Abbreviation used for naming saved model files and logs.
    """

    def __init__(
        self,
        img_size=28,
        patch_size=7,
        embed_dim=64,
        depth=6,
        n_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
        n_classes=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.abbreviation = "vit"

        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.cls_token = self.add_weight(
            shape=[1, 1, embed_dim], initializer="zeros", trainable=True
        )
        self.pos_embed = self.add_weight(
            shape=[1, 1 + self.patch_embed.n_patches, embed_dim],
            initializer="zeros",
            trainable=True,
        )

        self.pos_drop = layers.Dropout(p)

        self.blocks = [
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p,
            )
            for _ in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(n_classes, activation="softmax")

    def call(self, x):
        """
        Forward pass for the Vision Transformer.

        @Parameters:
        x : tf.Tensor
            Input tensor with shape (batch_size, img_size, img_size, channels).

        @Returns:
        tf.Tensor
            Output tensor with shape (batch_size, n_classes) representing class scores.
        """

        n_samples = tf.shape(x)[0]
        x = self.patch_embed(x)

        cls_tokens = tf.broadcast_to(
            self.cls_token, [n_samples, 1, self.embed_dim]
        )  # Add batch dimension to cls_token
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed[:, : tf.shape(x)[1], :]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x
