use burn::{
    module::{
        conv2d::Conv2d,
        embedding::Embedding,
        layernorm::LayerNorm,
        linear::Linear,
        module_trait::Module,
        positional_embedding::PositionEmbeddingRandom,
    },
    nn::activation::Activation,
    Backend, Device,
};

/// Encodes prompts for input to SAM's mask decoder.
#[derive(Module, Debug)]
pub struct PromptEncoder<B: Backend, A: Activation<B>, N: LayerNorm<B>> {
    embed_dim: usize,
    input_image_size: (usize, usize),
    image_embedding_size: (usize, usize),
    pe_layer: PositionEmbeddingRandom<B>,
    point_embeddings: Vec<Embedding<B>>,
    not_a_point_embed: Embedding<B>,
    mask_downscaling: MaskDownscaling<B, A, N>,
    no_mask_embed: Embedding<B>,
}

impl<B: Backend, A: Activation<B>, N: LayerNorm<B>> PromptEncoder<B, A, N> {
    pub fn new(
        embed_dim: usize,
        image_embedding_size: (usize, usize),
        input_image_size: (usize, usize),
        mask_in_chans: usize,
        activation: A,
        device: &Device<B>,
    ) -> Self {
        let pe_layer = PositionEmbeddingRandom::new(embed_dim / 2, device);
        let num_point_embeddings = 4; // pos/neg point + 2 box corners
        let point_embeddings = (0..num_point_embeddings)
            .map(|_| Embedding::new(1, embed_dim, device))
            .collect();
        let not_a_point_embed = Embedding::new(1, embed_dim, device);

        let mask_input_size = (4 * image_embedding_size.0, 4 * image_embedding_size.1);
        let mask_downscaling = MaskDownscaling::new(
            mask_in_chans,
            embed_dim,
            activation.clone(),
            mask_input_size,
            device,
        );
        let no_mask_embed = Embedding::new(1, embed_dim, device);

        Self {
            embed_dim,
            input_image_size,
            image_embedding_size,
            pe_layer,
            point_embeddings,
            not_a_point_embed,
            mask_downscaling,
            no_mask_embed,
        }
    }

    fn get_dense_pe(&self) -> Tensor<B, 4> {
        self.pe_layer
            .forward(self.image_embedding_size)
            .view_unbind([1, -1, self.image_embedding_size.0, self.image_embedding_size.1])
    }

    fn embed_points(
        &self,
        points: Tensor<B, 2>,
        labels: Tensor<B, 2>,
        pad: bool,
    ) -> Tensor<B, 3> {
        
        let points = points + 0.5; // Shift to center of pixel
        let mut point_embedding = self
            .pe_layer
            .forward_with_coords(points, self.input_image_size);

        if pad {
            let padding_point = Tensor::zeros(
                &[points.shape()[0], 1, 2],
                self.pe_layer.device(),
            );
            let padding_label = Tensor::ones(
                &[labels.shape()[0], 1],
                self.pe_layer.device(),
            ) * -1.0;
            point_embedding = Tensor::cat(&[point_embedding, padding_point], 2).unwrap();
            let labels = Tensor::cat(&[labels, padding_label], 1).unwrap();
        }

        point_embedding.index_fill_(
            &[labels == -1.0],
            self.not_a_point_embed.weight.view_unbind([1, -1]),
        );
        point_embedding.index_fill_(
            &[labels == 0.0],
            self.point_embeddings[0].weight.view_unbind([1, -1]),
        );
        point_embedding.index_fill_(
            &[labels == 1.0],
            self.point_embeddings[1].weight.view_unbind([1, -1]),
        );

        point_embedding
    }

    fn embed_boxes(&self, boxes: Tensor<B, 2>) -> Tensor<B, 3> {
        
        let boxes = boxes + 0.5; // Shift to center of pixel
        let coords = boxes.view_unbind([-1, 2, 2]);
        let mut corner_embedding = self
            .pe_layer
            .forward_with_coords(coords, self.input_image_size);

        corner_embedding.index_fill_(
            &[coords.new_ones(coords.shape(), coords.device(), false), 0],
            self.point_embeddings[2].weight.view_unbind([1, -1]),
        );
        corner_embedding.index_fill_(
            &[coords.new_ones(coords.shape(), coords.device(), false), 1],
            self.point_embeddings[3].weight.view_unbind([1, -1]),
        );

        corner_embedding
    }

    fn _embed_masks(&self, masks: Tensor<B, 4>) -> Tensor<B, 4> {
        let mask_embedding = self.mask_downscaling.forward(masks);
        mask_embedding
    }

    fn _get_batch_size(
        &self,
        points: Option<&(Tensor<B, 2>, Tensor<B, 1>)>,
        boxes: Option<&Tensor<B, 2>>,
        masks: Option<&Tensor<B, 4>>,
    ) -> usize {
        
        if let Some((coords, _)) = points {
            coords.size()[0]
        } else if let Some(boxes) = boxes {
            boxes.size()[0]
        } else if let Some(masks) = masks {
            masks.size()[0]
        } else {
            1
        }
    }

    fn _get_device(&self) -> Device<B> {
        self.point_embeddings[0].weight.device().clone()
    }

    pub fn forward(
        &self,
        points: Option<(Tensor<B, 2>, Tensor<B, 1>)>,
        boxes: Option<Tensor<B, 2>>,
        masks: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let bs = self._get_batch_size(points.as_ref(), boxes.as_ref(), masks.as_ref());
        let mut sparse_embeddings = Tensor::zeros(&[bs, 0, self.embed_dim], self._get_device());

        if let Some((coords, labels)) = points {
            let point_embeddings = self._embed_points(coords, labels, boxes.is_none());
            sparse_embeddings = sparse_embeddings.concat_dim(1, &point_embeddings);
        }

        if let Some(boxes) = boxes {
            let box_embeddings = self._embed_boxes(boxes);
            sparse_embeddings = sparse_embeddings.concat_dim(1, &box_embeddings);
        }

        let dense_embeddings = if let Some(masks) = masks {
            self._embed_masks(masks)
        } else {
            self.no_mask_embed.weight
                .reshape([1, -1, 1, 1])
                .expand([bs, -1, self.image_embedding_size.0, self.image_embedding_size.1])
        };

        (sparse_embeddings, dense_embeddings)
    }
}

