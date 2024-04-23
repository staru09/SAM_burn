use burn::tensor::Tensor;
use burn::module::{Linear, Module};
use burn::backend::Backend;
use burn::ops::{GeLU, LayerNorm};

#[derive(Module, Debug)]
pub struct MLPBlock<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
    act: GeLU,
}

impl<B: Backend> MLPBlock<B> {
    pub fn new(embedding_dim: usize, mlp_dim: usize) -> Self {
        let lin1 = Linear::new(embedding_dim, mlp_dim);
        let lin2 = Linear::new(mlp_dim, embedding_dim);
        let act = GeLU::new();

        Self { lin1, lin2, act }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.lin1.forward(x);
        let x = self.act.forward(x);
        self.lin2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm2d<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f32,
}

impl<B: Backend> LayerNorm2d<B> {
    pub fn new(num_channels: usize, eps: f32) -> Self {
        let weight = Tensor::ones(&[num_channels], &B::default());
        let bias = Tensor::zeros(&[num_channels], &B::default());

        Self {
            weight,
            bias,
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let u = x.mean(1, true);
        let s = (&x - &u).pow2().mean(1, true);
        let x = (&x - &u) / (&s + self.eps).sqrt();
        x * &self.weight.view([-1, 1, 1]) + &self.bias.view([-1, 1, 1])
    }
}