import numpy as np

CORRUPTIONS = [
  'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
  'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
  'brightness', 'contrast', 'elastic_transform', 'pixelate',
  'jpeg_compression'
]

ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def compute_mce(corruption_accs):
    """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
    mce = 0.
    for i in range(len(CORRUPTIONS)):
        avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
        ce = 100 * avg_err / ALEXNET_ERR[i]
        mce += ce / 15
    return mce


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
    b = args.batch_size / 256.
    k = args.epochs // 3
    if epoch < k:
      m = 1
    elif epoch < 2 * k:
        m = 0.1
    else:
        m = 0.01
    lr = args.learning_rate * m * b
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr