#[derive(Module, Debug)]
pub struct TwoWayAttentionBlock<B: Backend> {
    self_attn: Attention<B>,
    norm1: LayerNorm<B, 2>,
    cross_attn_token_to_image: Attention<B>,
    norm2: LayerNorm<B, 2>,
    mlp: MLPBlock<B>,
    norm3: LayerNorm<B, 2>,
    norm4: LayerNorm<B, 2>,
    cross_attn_image_to_token: Attention<B>,
    skip_first_layer_pe: bool,
}

impl<B: Backend> TwoWayAttentionBlock<B> {
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        activation: ReLU,
        attention_downsample_rate: usize,
        skip_first_layer_pe: bool,
    ) -> Self {
        let self_attn = Attention::new(embedding_dim, num_heads, 1);
        let norm1 = LayerNorm::new(embedding_dim);
        let cross_attn_token_to_image =
            Attention::new(embedding_dim, num_heads, attention_downsample_rate);
        let norm2 = LayerNorm::new(embedding_dim);
        let mlp = MLPBlock::new(embedding_dim, mlp_dim, activation);
        let norm3 = LayerNorm::new(embedding_dim);
        let norm4 = LayerNorm::new(embedding_dim);
        let cross_attn_image_to_token =
            Attention::new(embedding_dim, num_heads, attention_downsample_rate);

        Self {
            self_attn,
            norm1,
            cross_attn_token_to_image,
            norm2,
            mlp,
            norm3,
            norm4,
            cross_attn_image_to_token,
            skip_first_layer_pe,
        }
    }

    pub fn forward(
        &self,
        queries: Tensor<B, 3>,
        keys: Tensor<B, 3>,
        query_pe: Tensor<B, 3>,
        key_pe: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)
        } else {
            let q = queries + query_pe;
            let attn_out = self.self_attn.forward(q, q, queries);
            queries + attn_out
        };
        queries = self.norm1.forward(queries);
        let q = queries + query_pe;
        let k = keys + key_pe;
        let attn_out = self.cross_attn_token_to_image.forward(q, k, keys);
        queries = queries + attn_out;
        queries = self.norm2.forward(queries);

        let mlp_out = self.mlp.forward(queries);
        queries = queries + mlp_out;
        queries = self.norm3.forward(queries);

        let q = queries + query_pe;
        let k = keys + key_pe;
        let attn_out = self.cross_attn_image_to_token.forward(k, q, queries);
        let mut keys = keys + attn_out;
        keys = self.norm4.forward(keys);

        (queries, keys)
    }
}