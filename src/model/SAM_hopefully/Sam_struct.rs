use burn::tensor::Tensor;
use burn::module::Module;
use burn::backend::Backend;

#[derive(Debug, Module)]
pub struct Sam<B: Backend> {
    pub image_encoder: ImageEncoderViT,
    pub prompt_encoder: PromptEncoder,
    pub mask_decoder: MaskDecoder,
    pub pixel_mean: Tensor<B, 3>,
    pub pixel_std: Tensor<B, 3>,
    pub mask_threshold: f32,
    pub image_format: String,
}

impl<B: Backend> Sam<B> {
    pub fn new(
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: [f32; 3],
        pixel_std: [f32; 3],
        mask_threshold: f32,
        image_format: String,
    ) -> Self {
        let pixel_mean = Tensor::from_vec(&pixel_mean, &[3, 1, 1], &B::default());
        let pixel_std = Tensor::from_vec(&pixel_std, &[3, 1, 1], &B::default());

        Sam {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            pixel_mean,
            pixel_std,
            mask_threshold,
            image_format,
        }
    }

    pub fn device(&self) -> &Device<B> {
        self.pixel_mean.device()
    }
    
    pub fn forward(
        &self,
        batched_input: Vec<HashMap<String, Tensor<B, 4>>>,
        multimask_output: bool,
    ) -> Vec<HashMap<String, Tensor<B, 4>>> {
        let input_images = Tensor::stack(
            &batched_input
                .iter()
                .map(|x| self.preprocess(x["image"].clone()))
                .collect::<Vec<_>>(),
            0,
        );

        let image_embeddings = self.image_encoder.forward(input_images);

        let mut outputs = Vec::new();

        for (image_record, curr_embedding) in
            batched_input.iter().zip(image_embeddings.unbind(0))
        {
            let (points, sparse_embeddings, dense_embeddings) = if image_record.contains_key("point_coords") {
                let points = (
                    image_record["point_coords"].clone(),
                    image_record["point_labels"].clone(),
                );
                let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                    Some(points),
                    image_record.get("boxes").cloned(),
                    image_record.get("mask_inputs").cloned(),
                );
                (Some(points), sparse_embeddings, dense_embeddings)
            } else {
                let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                    None,
                    image_record.get("boxes").cloned(),
                    image_record.get("mask_inputs").cloned(),
                );
                (None, sparse_embeddings, dense_embeddings)
            };

            let (low_res_masks, iou_predictions) = self.mask_decoder.forward(
                curr_embedding.unsqueeze(0),
                self.prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
            );

            let masks = self.postprocess_masks(
                low_res_masks,
                image_record["image"].size()[2..],
                image_record["original_size"].into(),
            );

            let masks = masks.gt(&self.mask_threshold);

            outputs.push(
                hashmap! {
                    "masks".to_string() => masks,
                    "iou_predictions".to_string() => iou_predictions,
                    "low_res_logits".to_string() => low_res_masks,
                }
            );
        }

        outputs
    }
}