#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    layers: Vec<Linear<B>>,
    sigmoid_output: bool,
}

impl<B: Backend> MLP<B> {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: bool,
        device: &Device<B>,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        let mut prev_dim = input_dim;

        for _ in 0..num_layers - 1 {
            let linear = LinearConfig::new(prev_dim, hidden_dim).init(device);
            layers.push(linear);
            prev_dim = hidden_dim;
        }

        let output_layer = LinearConfig::new(prev_dim, output_dim).init(device);
        layers.push(output_layer);

        Self {
            layers,
            sigmoid_output,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        let num_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);

            if i < num_layers - 1 {
                x = Relu::new().forward(x);
            } else if self.sigmoid_output {
                x = Sigmoid::new().forward(x);
            }
        }

        x
    }
}