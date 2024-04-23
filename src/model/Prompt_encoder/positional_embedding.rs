#[derive(Module, Debug)]
pub struct PositionEmbeddingRandom<B: burn::Backend> {
    positional_encoding_gaussian_matrix: Array2<B::Scalar>,
    _backend: PhantomData<B>,
}

impl<B: burn::Backend> PositionEmbeddingRandom<B> {
    pub fn new(num_pos_feats: usize, scale: Option<B::Scalar>, device: &Device<B>) -> Self {
        let scale = scale.unwrap_or_else(|| B::Scalar::one());
        let positional_encoding_gaussian_matrix =
            Array2::from_shape_fn((2, num_pos_feats), |_| B::Scalar::random());
        positional_encoding_gaussian_matrix.mul_scalar(scale);

        Self {
            positional_encoding_gaussian_matrix,
            _backend: PhantomData,
        }
    }

    fn _pe_encoding(&self, coords: &Tensor<B, 3>) -> Tensor<B, 3> {
        let mut coords = coords.clone();
        coords.mul_scalar(B::Scalar::from_f64(2.0));
        coords.sub_scalar(B::Scalar::one());
        let coords = coords.matmul(&self.positional_encoding_gaussian_matrix.view(2, -1));
        let coords = coords.mul_scalar(B::Scalar::from_f64(2.0 * std::f64::consts::PI));
        
        concat(
            &[&coords.sin(), &coords.cos()],
            coords.ndim() - 1,
            false,
        )
    }

    pub fn forward(&self, size: (usize, usize)) -> Tensor<B, 3> {
        let (h, w) = size;
        let device = self.positional_encoding_gaussian_matrix.device();

        let grid = Tensor::<B, 2>::ones(&[h, w], device);
        let mut y_embed = grid.cumsum(0, false);
        y_embed.sub_scalar(B::Scalar::from_f64(0.5));
        let mut x_embed = grid.cumsum(1, false);
        x_embed.sub_scalar(B::Scalar::from_f64(0.5));

        y_embed.div_scalar(B::Scalar::from_f64(h as f64));
        x_embed.div_scalar(B::Scalar::from_f64(w as f64));

        let coords = Tensor::stack(&[&x_embed, &y_embed], -1);
        let pe = self._pe_encoding(&coords);

        pe.permute(&[2, 0, 1])
    }

    pub fn forward_with_coords(
        &self,
        coords_input: &Tensor<B, 3>,
        image_size: (usize, usize),
    ) -> Tensor<B, 3> {
        let (h, w) = image_size;
        let mut coords = coords_input.clone();

        coords.index_select_mut(2, &Tensor::<B, 1>::from(vec![0])).div_scalar(B::Scalar::from_f64(w as f64));
        coords.index_select_mut(2, &Tensor::<B, 1>::from(vec![1])).div_scalar(B::Scalar::from_f64(h as f64));

        self._pe_encoding(&coords)
    }
}