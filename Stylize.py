from StyleTransfer import StyleTransfer, ShallowAutoencoder
from PIL import Image
import os

content_imgs = os.listdir("sample/content_img/")
style_imgs = os.listdir("sample/style_img/")

i = 1
for content_img in content_imgs :
    for style_img in style_imgs :
        Ic = Image.open("sample/content_img/" + content_img).convert('RGB')
        Is = Image.open("sample/style_img/" + style_img).convert('RGB')
        
        final = StyleTransfer(Ic, Is, 0.7)
        final.save("sample/style_transfer/" + str(i) + ".jpg")
        
        i += 1