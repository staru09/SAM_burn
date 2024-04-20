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
        x.swap_dims(1, 3)
    }
}


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

        // qkv with shape (3, B, num_heads, hw, head_dim)
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
        let x = attn @ v;
        let x = x
            .reshape([b, self.num_heads, h, w, -1])
            .swap_dims(1, 3)
            .swap_dims(2, 3)
            .flatten(1, 3);

        self.proj.forward(x)
    }
}

use burn::{module::Module, nn::{Linear, LinearConfig, Relu}};

#[derive(Module, Debug)]
pub struct MLPBlock<B: Backend> {
    fc1: Linear<B>,
    act: Relu,
    fc2: Linear<B>,
}

impl<B: Backend> MLPBlock<B> {
    pub fn new(embedding_dim: usize, mlp_dim: usize, act: Relu, device: &Device<B>) -> Self {
        let fc1 = LinearConfig::new(embedding_dim, mlp_dim).init(device);
        let fc2 = LinearConfig::new(mlp_dim, embedding_dim).init(device);

        Self { fc1, act, fc2 }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.act.forward(x);
        self.fc2.forward(x)
    }
}

use burn::{tensor::Tensor, Backend};

pub fn window_partition<B: Backend>(
    input: Tensor<B, 4>,
    window_size: usize,
) -> (Tensor<B, 4>, (usize, usize)) {
    let b, h, w, c = input.shape().into();

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let hp = h + pad_h;
    let wp = w + pad_w;

    let padded = if pad_h > 0 || pad_w > 0 {
        input.pad([0, 0, 0, pad_w, 0, pad_h])
    } else {
        input
    };

    let windows = padded
        .reshape([b, hp / window_size, window_size, wp / window_size, window_size, c])
        .swap_dims(2, 3)
        .flatten(1, 4);

    (windows, (hp, wp))
}

use burn::{tensor::Tensor, Backend};

pub fn window_unpartition<B: Backend>(
    windows: Tensor<B, 4>,
    window_size: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Tensor<B, 4> {
    let b = windows.shape()[0] / (pad_hw.0 * pad_hw.1 / window_size / window_size);
    let hp = pad_hw.0;
    let wp = pad_hw.1;
    let h = hw.0;
    let w = hw.1;

    let x = windows
        .reshape([b, hp / window_size, wp / window_size, window_size, window_size, -1])
        .swap_dims(2, 3)
        .flatten(1, 4);

    if hp > h || wp > w {
        x.slice(1, 0, h).slice(2, 0, w)
    } else {
        x
    }
}

use burn::{tensor::Tensor, Backend};
use burn::nn::functional as F;

pub fn get_rel_pos<B: Backend>(
    q_size: usize,
    k_size: usize,
    rel_pos: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;

    let rel_pos_resized = if rel_pos.shape()[0] != max_rel_dist {
        F::interpolate(
            &rel_pos.reshape([1, rel_pos.shape()[0], -1]).swap_dims(1, 2),
            [max_rel_dist, rel_pos.shape()[1]],
            F::InterpolateMode::Linear,
            None,
            None,
            None,
            None,
        )
        .reshape([max_rel_dist, -1])
        .transpose(-1, -2)
    } else {
        rel_pos
    };

    let q_coords = Tensor::arange(q_size, 0, device)
        .view([q_size, 1])
        .cast::<f32>()
        * (k_size as f32 / q_size as f32).max(1.0);
    let k_coords = Tensor::arange(k_size, 0, device)
        .view([1, k_size])
        .cast::<f32>()
        * (q_size as f32 / k_size as f32).max(1.0);
    let relative_coords = q_coords - k_coords + (k_size - 1) as f32 * (q_size as f32 / k_size as f32).max(1.0);

    rel_pos_resized.index(&relative_coords.cast::<i64>().view([-1]))
}

use burn::{tensor::Tensor, Backend};

pub fn add_decomposed_rel_pos<B: Backend>(
    attn: Tensor<B, 3>,
    q: Tensor<B, 3>,
    rel_pos_h: Tensor<B, 2>,
    rel_pos_w: Tensor<B, 2>,
    q_size: (usize, usize),
    k_size: (usize, usize),
) -> Tensor<B, 3> {
    let b, q_h, q_w, _ = q.shape().into();
    let k_h = k_size.0;
    let k_w = k_size.1;

    let rh = get_rel_pos(q_h, k_h, rel_pos_h);
    let rw = get_rel_pos(q_w, k_w, rel_pos_w);

    let r_q = q.reshape([b, q_h, q_w, -1]);
    let rel_h = (r_q * rh.view([1, q_h, 1, -1])).sum(-1);
    let rel_w = (r_q * rw.view([1, 1, q_w, -1])).sum(-1);

    attn.reshape([b, q_h, q_w, k_h, k_w])
        + rel_h.unsqueeze(-2)
        + rel_w.unsqueeze(-1)
        .flatten(1, 3)
}