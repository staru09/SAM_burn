use burn::tensor::Tensor;
use burn::module::{Linear, Module};
use burn::backend::Backend;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    internal_dim: usize,
    num_heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(embedding_dim: usize, num_heads: usize, downsample_rate: usize) -> Self {
        let internal_dim = embedding_dim / downsample_rate;
        assert!(
            internal_dim % num_heads == 0,
            "num_heads must divide embedding_dim."
        );

        let q_proj = Linear::new(embedding_dim, internal_dim);
        let k_proj = Linear::new(embedding_dim, internal_dim);
        let v_proj = Linear::new(embedding_dim, internal_dim);
        let out_proj = Linear::new(internal_dim, embedding_dim);

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            internal_dim,
            num_heads,
        }
    }

    fn _separate_heads(&self, x: Tensor<B, 3>, num_heads: usize) -> Tensor<B, 4> {
        let b = x.size(0);
        let n = x.size(1);
        let c = x.size(2);
        let c_per_head = c / num_heads;

        x.reshape(&[b, n, num_heads, c_per_head])
            .permute(&[0, 2, 1, 3])
    }

    fn _recombine_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let b = x.size(0);
        let n_heads = x.size(1);
        let n_tokens = x.size(2);
        let c_per_head = x.size(3);

        x.permute(&[0, 2, 1, 3])
            .reshape(&[b, n_tokens, n_heads * c_per_head])
    }

    pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        // Input projections
        let q = self.q_proj.forward(q);
        let k = self.k_proj.forward(k);
        let v = self.v_proj.forward(v);

        // Separate into heads
        let q = self._separate_heads(q, self.num_heads);
        let k = self._separate_heads(k, self.num_heads);
        let v = self._separate_heads(v, self.num_heads);

        // Attention
        let _, _, _, c_per_head = q.size();
        let attn = q.matmul(&k.permute(&[0, 1, 3, 2])) / (c_per_head as f32).sqrt();
        let attn = attn.softmax(-1);

        // Get output
        let out = attn.matmul(&v);
        let out = self._recombine_heads(out);
        self.out_proj.forward(out)
    }
}