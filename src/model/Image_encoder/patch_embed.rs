#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        in_chans: usize,
        embed_dim: usize,
        device: &Device<B>,
    ) -> Self {
        let proj = Conv2dConfig::new([in_chans, embed_dim], kernel_size)
            .with_stride(stride)
            .with_padding(PaddingConfig2d::Explicit(padding.0, padding.1))
            .init(device);

        Self { proj }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.proj.forward(input);
        //(0,1,2,3)->(0,2,3,1)
        x.swap_dims(1,2);
        x.swap_dims(1,3)
    }
}