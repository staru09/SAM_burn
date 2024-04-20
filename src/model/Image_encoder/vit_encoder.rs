use burn::{
    module::Module,
    nn::{
        Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Parameter,
        PaddingConfig2d, Relu,
    },
    Device, Tensor,
};
use std::marker::PhantomData;

#[derive(Module, Debug)]
pub struct ImageEncoderViT<B: Backend> {
    patch_embed: PatchEmbed<B>,
    pos_embed: Option<Parameter<B, 4>>,
    blocks: Vec<Block<B>>,
    neck: Neck<B>,
}

impl<B: Backend> ImageEncoderViT<B> {
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f32,
        out_chans: usize,
        qkv_bias: bool,
        norm_layer: LayerNorm<B, 2>,
        act_layer: Relu,
        use_abs_pos: bool,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        window_size: usize,
        global_attn_indexes: &[usize],
        device: &Device<B>,
    ) -> Self {
        let patch_embed = PatchEmbed::new(
            (patch_size, patch_size),
            (patch_size, patch_size),
            (0, 0),
            in_chans,
            embed_dim,
            device,
        );

        let pos_embed = if use_abs_pos {
            let pos_embed = Tensor::zeros(
                [1, img_size / patch_size, img_size / patch_size, embed_dim],
                device,
            );
            Some(Parameter::new(pos_embed))
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let window_size = if global_attn_indexes.contains(&i) {
                0
            } else {
                window_size
            };

            let block = Block::new(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                norm_layer.clone(),
                act_layer.clone(),
                use_rel_pos,
                rel_pos_zero_init,
                window_size,
                (img_size / patch_size, img_size / patch_size),
                device,
            );
            blocks.push(block);
        }

        let neck = Neck::new(embed_dim, out_chans, device);

        Self {
            patch_embed,
            pos_embed,
            blocks,
            neck,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.patch_embed.forward(input);
        if let Some(pos_embed) = &self.pos_embed {
            x = x + pos_embed;
        }

        for block in &self.blocks {
            x = block.forward(x);
        }

        self.neck.forward(x.swap_dims(1, 3))
    }
}

#[derive(Module, Debug)]
pub struct Neck<B: Backend> {
    conv1: Conv2d<B>,
    norm1: LayerNorm<B, 2>,
    conv2: Conv2d<B>,
    norm2: LayerNorm<B, 2>,
}

impl<B: Backend> Neck<B> {
    pub fn new(in_chans: usize, out_chans: usize, device: &Device<B>) -> Self {
        let conv1 = Conv2dConfig::new([in_chans, out_chans], [1, 1])
            .with_bias(false)
            .init(device);
        let norm1 = LayerNormConfig::new(out_chans).init(device);
        let conv2 = Conv2dConfig::new([out_chans, out_chans], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .init(device);
        let norm2 = LayerNormConfig::new(out_chans).init(device);

        Self {
            conv1,
            norm1,
            conv2,
            norm2,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input);
        let x = self.norm1.forward(x);
        let x = self.conv2.forward(x);
        self.norm2.forward(x)
    }
}