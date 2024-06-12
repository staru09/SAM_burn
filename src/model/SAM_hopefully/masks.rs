use burn::tensor::Tensor;
use burn::backend::Backend;

impl<B: Backend> YourModel<B> {
    pub fn postprocess_masks(
        &self,
        masks: Tensor<B, 4>,
        input_size: (usize, usize),
        original_size: (usize, usize),
    ) -> Tensor<B, 4> {
        let img_size = self.image_encoder.img_size;
        let masks = Interpolate::new(
            masks,
            &[img_size, img_size],
            None,
            Interpolate::Mode::Bilinear,
            false,
        );

        let (input_h, input_w) = input_size;
        let masks = masks.slice(2, 0, input_h, None);
        let masks = masks.slice(3, 0, input_w, None);

        let (original_h, original_w) = original_size;
        Interpolate::new(
            masks,
            &[original_h, original_w],
            None,
            Interpolate::Mode::Bilinear,
            false,
        )
    }

    pub fn preprocess(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (h, w) = (x.size(2), x.size(3));
        let padh = self.image_encoder.img_size - h;
        let padw = self.image_encoder.img_size - w;

        let x = (x - &self.pixel_mean) / &self.pixel_std;
        Pad::new(x, &[0, padw, 0, padh])
    }
}