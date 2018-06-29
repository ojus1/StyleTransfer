Neural Style transfer based on the paper "Universal Style Transfer via Feature Transforms" https://arxiv.org/pdf/1705.08086.pdf


Used a small two-layer-deep autoencoder, that is trained on a subset of COCO dataset.

Results are very bad, as the autoencoder isn't deep enough to learn complex features, and the image size is very small at 64x64. This is due to lack of a decent NVIDIA GPU and horrendous training speed of a laptop CPU. Some outputs of the network are in the "sample/style_transfer/"

It currently is able to learn about color composition, and hence the results are just tinted with various colors. Check out the images used as "style image" at "sample/style_img/".

I'll be training a deeper autoencoder on a larger image size in the future. This project is very scalable.

Requirements to run the scripts are at requirements.txt
