import tkinter as tk
import framework
from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
# To get the dialog box to open when required  
from tkinter import filedialog 
class ImageViewer(framework.Framework):

    start_x, start_y = 0, 0
    end_x, end_y = 0, 0

    tool_bar_functions = (
        "train","eval"
    )
    canvas_names = (
        "Median_Nerve",
        "Flexor_Tendons",
        "Carpal_Tunnel"
    )
    selected_tool_bar_function = tool_bar_functions[0]

    def create_tool_bar_buttons(self):
        for index, name in enumerate(self.tool_bar_functions):
            icon = tk.PhotoImage(file='icons/' + name + '.png')
            self.button = tk.Button(
                self.tool_bar, image=icon, command=lambda index=index: self.on_tool_bar_button_clicked(index))
            self.button.grid(
                row=index // 2, column=1 + index % 2, sticky='nsew')
            self.button.image = icon

    def on_tool_bar_button_clicked(self, button_index):
        self.selected_tool_bar_function = self.tool_bar_functions[button_index]
        self.remove_options_from_top_bar()
        self.display_options_in_the_top_bar()

    def display_options_in_the_top_bar(self):
        self.show_selected_tool_icon_in_top_bar(
            self.selected_tool_bar_function)

    def remove_options_from_top_bar(self):
        for child in self.top_bar.winfo_children():
            child.destroy()

    def show_selected_tool_icon_in_top_bar(self, function_name):
        display_name = function_name.replace("_", " ").capitalize() + ":"
        tk.Label(self.top_bar, text=display_name).pack(side="left")
        photo = tk.PhotoImage(
            file='icons/' + function_name + '.png')
        label = tk.Label(self.top_bar, image=photo)
        label.image = photo
        label.pack(side="left")

    def on_mouse_button_pressed(self, event):
        self.start_x = self.end_x = self.canvas.canvasx(event.x)
        self.start_y = self.end_y = self.canvas.canvasy(event.y)

    def on_mouse_button_pressed_motion(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)

    def on_mouse_button_released(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)

    def on_mouse_unpressed_motion(self, event):
        pass

    def __init__(self, root):
        super().__init__(root)
        root.title("GUI")
        self.create_gui()

    def create_gui(self):
        self.create_menu()
        self.create_top_bar()
        self.create_tool_bar()
        self.create_tool_bar_buttons()
        self.create_canvas(text = "test")
        self.bind_menu_accelrator_keys()

    def create_menu(self):
        self.menubar = tk.Menu(self.root)
        menu_definitions = (
            'File- &T1Seq/Ctrl+1/self.select_MN_GT_image, \
                &T2Seq/Ctrl+2/self.select_FT_GT_image,\
                &GTSeq/Ctrl+G/self.select_CT_GT_image, sep, Exit/Alt+F4/self.on_close_menu_clicked',
            'Edit- Undo/Ctrl+Z/self.on_undo_menu_clicked, sep',
            'View- Zoom in//self.on_canvas_zoom_in_menu_clicked,Zoom Out//self.on_canvas_zoom_out_menu_clicked',
            'About- About/F1/self.on_about_menu_clicked'
        )
        self.build_menu(menu_definitions)

    def create_top_bar(self):
        self.top_bar = tk.Frame(self.root, height=25, relief="raised")
        self.top_bar.pack(fill="x", side="top", pady=2)

    def create_tool_bar(self):
        self.tool_bar = tk.Frame(self.root, relief="raised", width=50)
        self.tool_bar.pack(fill="y", side="left", pady=3)

    def create_canvas(self, text, width_param = 225, height_param = 225): 
        self.Median_Nerve_image = ImageTk.PhotoImage(
            Image.open('../data/0/MN/0.jpg').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Median_Nerve_label = tk.Label(
            root,text='../data/0/MN/0.jpg',
            image = self.Median_Nerve_image, 
            compound = 'top'
        )
        self.Median_Nerve_label.pack(side=TOP)

        self.Flexor_Tendons_image = ImageTk.PhotoImage(
            Image.open('../data/0/FT/0.jpg').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Flexor_Tendons_label = tk.Label(
            root,text='../data/0/FT/0.jpg',
            image = self.Flexor_Tendons_image, 
            compound = 'top'
        )
        self.Flexor_Tendons_label.pack(side=TOP)

        self.Carpal_tunnel_image = ImageTk.PhotoImage(
            Image.open('../data/0/CT/0.jpg').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Carpal_tunnel_label = tk.Label(
            root,text='../data/0/CT/0.jpg',
            image = self.Carpal_tunnel_image, 
            compound = 'top'
        )
        self.Carpal_tunnel_label.pack(side = BOTTOM)
        
    def create_scroll_bar(self):
        x_scroll = tk.Scrollbar(self.canvas_frame, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")
        x_scroll.config(command=self.canvas.xview)
        y_scroll = tk.Scrollbar(self.canvas_frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")
        y_scroll.config(command=self.canvas.yview)
        self.canvas.config(
            xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

    def bind_menu_accelrator_keys(self):
        self.root.bind('<KeyPress-F1>', self.on_about_menu_clicked)
        self.root.bind('<Control-s>', self.on_save_menu_clicked)
        self.root.bind('<Control-S>', self.on_save_menu_clicked)
        self.root.bind('<Control-z>', self.on_undo_menu_clicked)
        self.root.bind('<Control-Z>', self.on_undo_menu_clicked)
    @classmethod
    def open_image(cls, filename, width = 224, height = 224):
        img = ImageTk.PhotoImage(
            Image.open(filename).resize(
                (width, height), Image.ANTIALIAS)
        )
        return img
    def select_MN_GT_image(self, event = None):
        filename = filedialog.askopenfilename(title ='files browser')
        self.Median_Nerve_image = ImageViewer.open_image(filename)
        self.Median_Nerve_label.configure(image = self.Median_Nerve_image)
        self.Median_Nerve_label.image = self.Median_Nerve_image
        self.Median_Nerve_label.pack(side = 'top')
        self.Median_Nerve_label.configure(text = "Median_Nerve:" + filename)

    def select_FT_GT_image(self, event = None):
        filename = filedialog.askopenfilename(title ='files browser')
        self.Flexor_Tendons_image = ImageViewer.open_image(filename)
        self.Flexor_Tendons_label.configure(image = self.Flexor_Tendons_image)
        self.Flexor_Tendons_label.image = self.Flexor_Tendons_image
        self.Flexor_Tendons_label.pack(side = 'top')
        self.Flexor_Tendons_label.configure(text = "Flexor_Tendons:" + filename)
        
    def select_CT_GT_image(self, event = None):
        filename = filedialog.askopenfilename(title ='files browser')
        self.Carpal_tunnel_image = ImageViewer.open_image(filename)
        self.Carpal_tunnel_label.configure(image = self.Carpal_tunnel_image)
        self.Carpal_tunnel_label.image = self.Carpal_tunnel_image
        self.Carpal_tunnel_label.pack(side = BOTTOM)
        self.Carpal_tunnel_label.configure(text = "Carpal_tunnel:" + filename)

    def on_save_menu_clicked(self, event=None):
        pass

    def on_save_as_menu_clicked(self):
        pass

    def on_canvas_zoom_out_menu_clicked(self):
        pass

    def on_canvas_zoom_in_menu_clicked(self):
        pass

    def on_close_menu_clicked(self):
        pass

    def on_undo_menu_clicked(self, event=None):
        pass

    def on_about_menu_clicked(self, event=None):
        pass



if __name__ == '__main__':
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
