import StyleTransfer
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

        self.img_shape = StyleTransfer.img_shape
        self.ids.capImage.size = self.img_shape
        self.ids.styleImage.size = self.img_shape
        self.ids.camera.resolution = self.img_shape

        self.content_img_path = 'out/captures/IMG_20180617_200922.png'
        self.style_img_path = ''
        
        self.ids.capImage.source = self.content_img_path
        self.ids.styleImage.source = self.content_img_path

    def ShowStylizedImage(self):
        pass
    
    def Capture(self):
        pass

class StylizeApp(App):
    def build(self):
        return Screen()

if __name__ == '__main__':
    StylizeApp().run()