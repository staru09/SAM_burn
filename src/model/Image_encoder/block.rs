/// Transformer block with support for window attention and residual propagation.
#[derive(BurnModule, Debug)]
pub struct Block<B: Backend, A: Activation<B>, N: LayerNorm<B>> {
    norm1: N,
    attn: Attention<B>,
    norm2: N,
    mlp: MLPBlock<B, A>,
    window_size: usize,
}

impl<B: Backend, A: Activation<B>, N: LayerNorm<B>> Block<B, A, N> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        norm_layer: N,
        act_layer: A,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        window_size: usize,
        input_size: Option<(usize, usize)>,
        device: &Device<B>,
    ) -> Self {
        let norm1 = norm_layer.clone();
        let attn = Attention::new(
            dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            rel_pos_zero_init,
            input_size.unwrap_or((window_size, window_size)),
            device,
        );
        let norm2 = norm_layer;
        let mlp = MLPBlock::new(dim, (dim as f32 * mlp_ratio) as usize, act_layer, device);

        Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shortcut = input.clone();
        let mut x = self.norm1.forward(input);

        // Window partition
        if self.window_size > 0 {
            let (h, w) = (x.shape()[1], x.shape()[2]);
            let (x, pad_hw) = window_partition(x, self.window_size);
            x = self.attn.forward(x);
            // Reverse window partition
            x = window_unpartition(x, self.window_size, pad_hw, (h, w));
        } else {
            x = self.attn.forward(x);
        }

        x = shortcut + x;
        x = x + self.mlp.forward(self.norm2.forward(x));

        x
    }
}