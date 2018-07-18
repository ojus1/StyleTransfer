#StyleTransfer
Neural Style Transfer using convolutional autoencoders.

This is an implementation of the paper "Universal Style Transfer using Feature Transforms" https://arxiv.org/pdf/1705.08086.pdf


Some results:



![Plus](sample/style_imgs/original/LOVE-BY-THE-PALETTE.jpg)
![This](sample/content_imgs/original/pexels-photo-1004684.jpeg)
![gives this](sample/examples/1.jpeg)
![This](sample/examples/11.jpeg)
![gives this](sample/examples/7.jpeg)
![This](sample/content_imgs/original/IMG_20180628_222617.jpg)
![gives this](sample/exampleslvl3.jpeg)


Instructions for running:
Change the "StyleTransfer" function call in "test_stylize.py"(test_stylize.py)
Currently there are three different autoencoders trained which are of varying depths/no. of filters,
Level1 being the shallowest, Level3 being the deepest.

