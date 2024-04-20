#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor<B, 2>>,
    rel_pos_w: Option<Tensor<B, 2>>,
    num_heads: usize,
    scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        input_size: Option<(usize, usize)>,
        device: &Device<B>,
    ) -> Self {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let qkv = LinearConfig::new(dim, dim * 3)
            .with_bias(qkv_bias)
            .init(device);
        let proj = LinearConfig::new(dim, dim).init(device);

        let (rel_pos_h, rel_pos_w) = if use_rel_pos {
            let input_size = input_size.expect("Input size must be provided if using relative positional encoding.");
            let rel_pos_h = Tensor::zeros([2 * input_size.0 - 1, head_dim], device);
            let rel_pos_w = Tensor::zeros([2 * input_size.1 - 1, head_dim], device);
            (Some(rel_pos_h), Some(rel_pos_w))
        } else {
            (None, None)
        };

        Self {
            qkv,
            proj,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
            num_heads,
            scale,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let b, h, w, _ = input.shape().into();
        let hw = h * w;

        // q, k, v with shape (3, B, num_heads, hw, head_dim)
        let qkv = self
            .qkv
            .forward(input.flatten(1, 3))
            .reshape([3, b, self.num_heads, hw, -1])
            .swap_dims(1, 2)
            .swap_dims(2, 3);

        // q, k, v with shape (B * num_heads, hw, head_dim)
        let (q, k, v) = qkv.unbind(0);
        let q = q.flatten(0, 1);
        let k = k.flatten(0, 1);
        let v = v.flatten(0, 1);

        let mut attn = q * self.scale * k.transpose(-1, -2);

        if self.use_rel_pos {
            attn = add_decomposed_rel_pos(
                attn,
                q,
                self.rel_pos_h.as_ref().unwrap(),
                self.rel_pos_w.as_ref().unwrap(),
                (h, w),
                (h, w),
            );
        }

        let attn = attn.softmax(-1);
        let x = attn.matmul(&v);
        let x = x
            .reshape([b, self.num_heads, h, w, -1])
            .swap_dims(1, 3)
            .swap_dims(2, 3)
            .flatten(1, 3);

        self.proj.forward(x)
    }
}