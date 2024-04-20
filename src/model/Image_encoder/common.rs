use burn::{
    module::Module,
    nn::activation::Activation,
    nn::{
        Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    nn::layernorm::LayerNorm,
    tensor_trait::Tensor,
    Backend, Device,
};

/// Multi-Layer Perceptron (MLP) Block.
#[derive(Module, Debug)]
pub struct MLPBlock<B: Backend, A: Activation<B>> {
    lin1: Linear<B>,
    lin2: Linear<B>,
    act: A,
}

impl<B: Backend, A: Activation<B>> MLPBlock<B, A> {
    pub fn new(
        embedding_dim: usize,
        mlp_dim: usize,
        act: A,
        device: &Device<B>,
    ) -> Self {
        let lin1 = Linear::new(embedding_dim, mlp_dim, device);
        let lin2 = Linear::new(mlp_dim, embedding_dim, device);

        Self { lin1, lin2, act }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.lin1.forward(input);
        let x = self.act.forward(x);
        self.lin2.forward(x)
    }
}

/// 2D Layer Normalization.
#[derive(Module, Debug)]
pub struct LayerNorm2d<B: Backend> {
    ln: LayerNorm<B, 2>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> LayerNorm2d<B> {
    pub fn new(num_channels: usize, eps: f32, device: &Device<B>) -> Self {
        let ln = LayerNormConfig::new(num_channels, eps).init(device);
        let weight = Tensor::ones(device, &[num_channels]);
        let bias = Tensor::zeros(device, &[num_channels]);

        Self { ln, weight, bias }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.ln.forward(input);
        let weight = self.weight.view_unbind([1, 1, -1]);
        let bias = self.bias.view_unbind([1, 1, -1]);

        x * weight + bias
    }
}