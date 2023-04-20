import numpy as np
import torch
from einops import parse_shape, rearrange


class RandomFilter(torch.nn.Module):
    def __init__(self, kernel_size, sigma, stochastic=False, sigma_min=0.):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.stochastic = stochastic
        if self.stochastic:
            self.kernels_size_candidates = torch.tensor([float(i) for i in range(self.kernel_size, self.kernel_size + 2, 2)])
            self.sigma_min = sigma_min
            self.sigma_max = sigma

    def forward(self, img):
        NUM_VIS = 10
        def imsave(img, idx, title, stage='3_rand_filter', gray=False):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)
            if gray:
                ax.imshow(img.cpu().detach().numpy(), cmap='gray')
            else:
                ax.imshow(img.cpu().detach().numpy())
            fig.tight_layout()
            # fig.savefig(f'/ws/data/dshong/prime/cifar/case{idx}.{stage}.{title}.png')
            plt.close(fig)

        if self.stochastic:
            self._sample_params()

        init_shape = img.shape
        if len(init_shape) < 4:
            img = rearrange(img, "c h w -> () c h w")

        shape_dict = parse_shape(img, "b c h w")
        batch_size = shape_dict["b"]
        img = rearrange(img, "b c h w -> c b h w")

        delta = torch.zeros((1, self.kernel_size, self.kernel_size), device=img.device)
        center = int(np.ceil(self.kernel_size / 2))
        delta[0, center, center] = 1.0

        conv_weight = rearrange(
            self.sigma * torch.randn((batch_size, self.kernel_size, self.kernel_size), device=img.device) + delta,
            "b h w -> b (h w)",
        )

        conv_weight = rearrange(conv_weight, "b (h w) -> b () h w", h=self.kernel_size)

        filtered_img = torch.nn.functional.conv2d(
            img, conv_weight, padding="same", groups=batch_size
        )

        # for i in range(NUM_VIS):
        #     for lev in range(3):
        #         imsave(img[:,i+128*lev,:,:].permute(1,2,0), i, f'0_img{lev}')
        #         imsave(conv_weight[i+128*lev, :, :, :].reshape(self.kernel_size, self.kernel_size), i, f'1_conv_weight{lev}', gray=True)
        #         _norm_filtered_img = filtered_img[:,i+128*lev,:,:]
        #         _ = _norm_filtered_img.reshape(3, -1)
        #         _min, _max = _.min(dim=-1).values, _.max(dim=-1).values
        #         _min = _min.reshape(-1, 1, 1).repeat(1, 32, 32)
        #         _max = _max.reshape(-1, 1, 1).repeat(1, 32, 32)
        #         _norm_filtered_img = (_norm_filtered_img - _min) / (_max - _min)
        #         imsave(_norm_filtered_img.permute(1,2,0), i, f'2_normalized_filtered_img{lev}')
        #         del _norm_filtered_img, _, _min, _max

        # Deal with NaN values due to mixed precision -> Convert them to 1.
        filtered_img[filtered_img.isnan()] = 1.

        filtered_img = rearrange(filtered_img, "c b h w -> b c h w")
        filtered_img = torch.clamp(filtered_img, 0., 1.).reshape(init_shape)

        # for i in range(NUM_VIS):
        #     for lev in range(3):
        #         imsave(filtered_img[i+128*lev].permute(1,2,0), i, f'3_filtered_img{lev}')

        return filtered_img

    def _sample_params(self):
        self.kernel_size = int(self.kernels_size_candidates[torch.multinomial(self.kernels_size_candidates, 1)].item())
        self.sigma = torch.FloatTensor([1]).uniform_(self.sigma_min, self.sigma_max).item()

    def __repr__(self):
        return self.__class__.__name__ + f"(sigma={self.sigma}, kernel_size={self.kernel_size})"


if __name__ == "__main__":
    import imagenet_stubs
    import matplotlib.pyplot as plt
    import PIL.Image
    import torchvision.transforms as T

    random_filter = RandomFilter(3, 1)

    for image_path in imagenet_stubs.get_image_paths():
        im = PIL.Image.open(image_path)
        plt.axis("off")
        plt.imshow(im, interpolation="nearest")
        plt.show()

        x = T.ToTensor()(im)
        # x = repeat(x, 'c h w -> b c h w', b=10)
        x2 = random_filter.forward(x)
        plt.imshow(x[0].permute(1, 2, 0))
        plt.savefig("test.png")

        plt.imshow(x2[0].permute(1, 2, 0))
        plt.savefig("test2.png")

        t = 0
