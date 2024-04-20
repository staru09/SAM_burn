// Window Partition
use burn::{tensor::Tensor, Backend};
use burn::nn::functional as F;

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

// Window Unpartition

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

// Relative Position

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

//Decomposed Relative Position

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