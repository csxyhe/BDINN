# Description

The complete code is coming soon!



## what is shown now!

**A plug-and-play banded deformable convolutional layer** significantly reduces the number of learnable parameters compared to standard deformable convolutional layers, providing **a more efficient and stable alternative** for training.
 
 The receptive field at each position is modulated by a rotation scalar, a stretch factor, and an optional re-scale factor, resulting in a banded shape as follows:

![pic](https://github.com/csxyhe/BDINN/blob/img/receptiveField.png)

**The implementation is fully based on PyTorch**, making it easy to integrate and use. Feel free to give it a try!

### How to use

- The module supports convolutional operations by utilizing **a non-square arrangement of sampling points**, allowing the `kernel_size` to be specified as either an integer or a tuple.

- `min_sscale`/`max_sscale`: A pair of threshold for the stretch scalar.
- `isRescale`: (bool) default to `True`.
  - For non-local perception capacity: `True` is recommend. In theory, the magnification factor for the sampling interval can range from $[1, \infty]$.
  - For stable training: `False` is recommend.


### Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
```

# Citation

```
@article{quan2025image,
  title={Image debanding using cross-scale invertible networks with banded deformable convolutions},
  author={Quan, Yuhui and He, Xuyi and Xu, Ruotao and Xu, Yong and Ji, Hui},
  journal={Neural Networks},
  volume={187},
  pages={107270},
  year={2025},
  publisher={Elsevier}
}
```
