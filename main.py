import Styletransfer
from Styletransfer import StyleTransfer, ShallowAutoencoder
import kivy
from glob import glob
from random import randint
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.camera import Camera
import time
import os
kivy.require('1.9.1')

class Screen(GridLayout):
    def __init__(self):
        super(Screen, self).__init__()
        self.rows = 5

        self.alpha = 0.5

        self.cap_img_name = ''
        self.cap_img_path = ''
        self.style_img_path = 'out/style_img/abstract-abstract-art-abstract-painting-935787.jpg'
        
    def ShowStylizedImage(self):
            try :
                target_img = StyleTransfer(self.content_img_path, self.style_img_path, alpha=self.alpha)
                target_img.save("out/stylized/" + self.cap_img_name)
                self.ids.capImage.source = "out/stylized/" + self.cap_img_name
            except :
                print("Error Occured")
                pass
            

    def Capture(self):
        try :
            camera = self.ids.camera
            timestr = time.strftime("%Y%m%d_%H%M%S")

            self.cap_img_name = "IMG_{}.png".format(timestr)
            camera.export_to_png("out/captures/" + self.cap_img_name)
            self.ids.capImage.source = "out/captures/" + self.cap_img_name
            self.content_img_path = "out/captures/" + self.cap_img_name
        except :
            print("Error occured")
            pass

class StylizeApp(App):
    def build(self):
        return Screen()

if __name__ == '__main__':
    StylizeApp().run()