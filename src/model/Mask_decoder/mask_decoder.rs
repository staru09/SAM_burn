use burn::{
    Device, Embedding, Linear, Module, ModuleList, Parameter, Sequential, Tensor, Transpose2d,
    TransposeConv2d,
};

#[derive(Module, Debug)]
pub struct MaskDecoder<B: Backend, A: Activation<B>> {
    transformer_dim: usize,
    transformer: ModuleImpl<B>,
    num_multimask_outputs: usize,
    iou_token: Embedding<B>,
    mask_tokens: Embedding<B>,
    output_upscaling: Sequential<B>,
    output_hypernetworks_mlps: ModuleList<B, MLP<B>>,
    iou_prediction_head: MLP<B>,
}

impl<B: Backend, A: Activation<B>> MaskDecoder<B, A> {
    pub fn new(
        transformer_dim: usize,
        transformer: ModuleImpl<B>,
        num_multimask_outputs: usize,
        activation: A,
        iou_head_depth: usize,
        iou_head_hidden_dim: usize,
        device: &Device<B>,
    ) -> Self {
        let iou_token = Embedding::new(1, transformer_dim, device);
        let num_mask_tokens = num_multimask_outputs + 1;
        let mask_tokens = Embedding::new(num_mask_tokens, transformer_dim, device);

        let output_upscaling = Sequential::new(
            vec![
                TransposeConv2dConfig::new([transformer_dim, transformer_dim / 4], [2, 2])
                    .with_strides([2, 2])
                    .init(device),
                LayerNorm2dConfig::new(transformer_dim / 4).init(device),
                activation,
                TransposeConv2dConfig::new([transformer_dim / 4, transformer_dim / 8], [2, 2])
                    .with_strides([2, 2])
                    .init(device),
                activation,
            ],
            device,
        );

        let output_hypernetworks_mlps = ModuleList::from_iter(
            (0..num_mask_tokens)
                .map(|_| {
                    MLP::new(
                        transformer_dim,
                        transformer_dim,
                        transformer_dim / 8,
                        3,
                        false,
                        device,
                    )
                })
                .collect::<Vec<_>>(),
            device,
        );

        let iou_prediction_head = MLP::new(
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth,
            false,
            device,
        );

        Self {
            transformer_dim,
            transformer,
            num_multimask_outputs,
            iou_token,
            mask_tokens,
            output_upscaling,
            output_hypernetworks_mlps,
            iou_prediction_head,
        }
    }
    pub fn forward(
        &self,
        image_embeddings: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        sparse_prompt_embeddings: Tensor<B, 3>,
        dense_prompt_embeddings: Tensor<B, 4>,
        multimask_output: bool,
    ) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let (masks, iou_pred) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        );

        let (masks, iou_pred) = if multimask_output {
            (masks.slice(1, None, 1), iou_pred.slice(1, None, 1))
        } else {
            (masks.slice(0, 1, 1), iou_pred.slice(0, 1, 1))
        };

        (masks, iou_pred)
    }

    fn predict_masks(
        &self,
        image_embeddings: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        sparse_prompt_embeddings: Tensor<B, 3>,
        dense_prompt_embeddings: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let output_tokens = Concat::concatenate(
            &[&self.iou_token.weight, &self.mask_tokens.weight],
            1,
        );
        let output_tokens = output_tokens
            .unsqueeze(0)
            .expand([sparse_prompt_embeddings.size()[0], -1, -1]);

        let tokens = Concat::concatenate(&[&output_tokens, &sparse_prompt_embeddings], 1);

        let src = Repeat::repeat(
            &image_embeddings,
            tokens.size()[0],
            Some(&[0]),
            Some(&[1, 2, 3]),
        );
        let src = src + dense_prompt_embeddings;
        let pos_src = Repeat::repeat(
            &image_pe,
            tokens.size()[0],
            Some(&[0]),
            Some(&[1, 2, 3]),
        );

        let (hs, src) = self.transformer.forward(src, pos_src, tokens);

        let iou_token_out = hs.slice(0, 1, 1);
        let mask_tokens_out = hs.slice(1, self.num_mask_tokens + 1, 1);

        let src = src.permute([0, 2, 3, 1]);
        let upscaled_embedding = self.output_upscaling.forward(src);

        let mut hyper_in_list = Vec::with_capacity(self.num_mask_tokens);
        for i in 0..self.num_mask_tokens {
            let hyper_in = self.output_hypernetworks_mlps[i].forward(mask_tokens_out.slice(i, i + 1, 2));
            hyper_in_list.push(hyper_in);
        }

        let hyper_in = Concat::concatenate(&hyper_in_list, 1);

        let (b, c, h, w) = upscaled_embedding.size();
        let masks = hyper_in.matmul(&upscaled_embedding.view([b, c, h * w]));
        let masks = masks.view([b, self.num_mask_tokens, h, w]);

        let iou_pred = self.iou_prediction_head.forward(iou_token_out);

        (masks, iou_pred)
    }
}