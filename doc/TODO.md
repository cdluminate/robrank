1. investigate the difference between original and alternative impls
   of pnp and amd defense.

2. implement new attack code for primitive amd attack and test

3. are the two projection steps different?

```python
#delta = th.clamp(images - images_orig, min=-self.eps, max=self.eps)
images = th.min(images, images_orig + self.eps)
images = th.max(images, images_orig - self.eps)
#images = th.clamp(images + delta, min=0., max=1.).detach()
images = th.clamp(images, min=0., max=1.)
images = images.clone().detach()
```
