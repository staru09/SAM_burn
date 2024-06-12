use burn::tensor::Tensor;
use burn::module::{Linear, Module, ModuleList};
use burn::backend::Backend;
use burn::ops::{Attention, LayerNorm, ReLU};

#[derive(Module, Debug)]
pub struct TwoWayTransformer<B: Backend> {
    layers: ModuleList<TwoWayAttentionBlock<B>>,
    final_attn_token_to_image: Attention<B>,
    norm_final_attn: LayerNorm<B, 2>,
}

impl<B: Backend> TwoWayTransformer<B> {
    pub fn new(
        depth: usize,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        attention_downsample_rate: usize,
    ) -> Self {
        let mut layers = ModuleList::new();
        for i in 0..depth {
            layers.push(TwoWayAttentionBlock::new(
                embedding_dim,
                num_heads,
                mlp_dim,
                ReLU::new(),
                attention_downsample_rate,
                i == 0,
            ));
        }

        let final_attn_token_to_image =
            Attention::new(embedding_dim, num_heads, attention_downsample_rate);
        let norm_final_attn = LayerNorm::new(embedding_dim);

        Self {
            layers,
            final_attn_token_to_image,
            norm_final_attn,
        }
    }

    pub fn forward(
        &self,
        image_embedding: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        point_embedding: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let bs = image_embedding.size(0);
        let c = image_embedding.size(1);
        let h = image_embedding.size(2);
        let w = image_embedding.size(3);

        let image_embedding = image_embedding.flatten(2).permute(&[0, 2, 1]);
        let image_pe = image_pe.flatten(2).permute(&[0, 2, 1]);

        // Prepare queries
        let mut queries = point_embedding;
        let mut keys = image_embedding;

        // Apply transformer blocks and final layernorm
        for layer in &self.layers {
            let (q, k) = layer.forward(
                queries,
                keys,
                point_embedding,
                image_pe,
            );
            queries = q;
            keys = k;
        }

        let q = queries + point_embedding;
        let k = keys + image_pe;

        let attn_out = self.final_attn_token_to_image.forward(q, k, keys);
        let queries = queries + attn_out;
        let queries = self.norm_final_attn.forward(queries);

        (queries, keys)
    }
}