import tkinter as tk
import tkinter.ttk
import framework
from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
# To get the dialog box to open when required  
from tkinter import filedialog 
import os
from collections import defaultdict
import functools
from predict import (predict, merge_pil_image)
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
            self.selected_tool_bar_function
        )
        options_function_name = "{}_options".format(self.selected_tool_bar_function)
        func = getattr(
            self, 
            options_function_name, 
            self.function_not_defined
        )
        func()

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

    def __init__(self, root, evaluation_model = None):
        super().__init__(root)
        root.title("GUI")
        self.T1_image_filename = '../data/0/T1/0.jpg'
        self.T2_image_filename = '../data/0/T2/0.jpg'
        self.Carpal_tunnel_filename = '..//data/0/CT/0.jpg'
        self.Flexor_Tendons_filename = '../data/0/FT/0.jpg'
        self.Median_Nerve_filename = '../data/0/MN/0.jpg'
        self.Sequence = defaultdict(list)
        self.Ouput_Sequence = defaultdict(list)
        self.scale_bar_slide_idx = 0
        self.create_gui()
        
    def create_gui(self):
        self.create_menu()
        self.create_top_bar()
        self.create_tool_bar()
        self.create_tool_bar_buttons()
        self.create_scale_bar()
        self.create_canvas(text = "test")
        self.bind_menu_accelrator_keys()

    def create_menu(self):
        self.menubar = tk.Menu(self.root)
        menu_definitions = (
            'File- &MNGroundTruth/Ctrl+1/self.select_MN_GT_image, \
                &FTGroundTruth/Ctrl+2/self.select_FT_GT_image,\
                &CTGroundTruth/Ctrl+G/self.select_CT_GT_image,\
                &T1Image/Ctrl+G/self.select_T1_image,\
                &T2Image/Ctrl+G/self.select_T2_image, \
                &Seq/Ctrl+G/self.select_sequence,sep, Exit/Alt+F4/self.on_close_menu_clicked',
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

    def create_scale_bar(self, range = 19.0):
        frame = tk.Frame(self.root)
        self.scale_bar = tk.ttk.Scale(
            frame, from_=0.0, to = range, command=self.on_scale_bar_clicked
        )
        self.scale_bar.set(0.0)
        self.scale_bar.pack()
        frame.pack()

    def create_canvas(self, text, width_param = 225, height_param = 225):
        #T1 image configuation
        self.T1_image = ImageTk.PhotoImage(
            Image.open('../data/0/T1/0.jpg').resize(
                (int(width_param * 1.5), int(height_param * 1.5)), Image.ANTIALIAS)
        )
        self.T1_image_label = tk.Label(
            root, text='../data/0/T1/0.jpg',anchor ='nw',
            image = self.T1_image, 
            compound = 'bottom'
        )
        self.T1_image_label.pack(side='left', anchor=NW)
        #T2 image configuation
        self.T2_image = ImageTk.PhotoImage(
            Image.open('../data/0/T2/0.jpg').resize(
                (int(width_param * 1.5), int(height_param * 1.5)), Image.ANTIALIAS)
        )
        self.T2_image_label = tk.Label(
            root, text='../data/0/T2/0.jpg',anchor ='nw',
            image = self.T2_image, 
            compound = 'bottom'
        )
        self.T2_image_label.pack(side='left', anchor=NW)

        #Median Nerve image configuation
        self.Median_Nerve_image = ImageTk.PhotoImage(
            Image.open('../data/0/MN/0.jpg').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Median_Nerve_label = tk.Label(
            root,text='Median_Nerve:../data/0/MN/0.jpg',anchor ='ne',
            image = self.Median_Nerve_image, 
            compound = 'top'
        )
        self.Median_Nerve_label.pack(side='left', anchor=NE)

        self.Flexor_Tendons_image = ImageTk.PhotoImage(
            Image.open('../data/0/FT/0.jpg').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Flexor_Tendons_label = tk.Label(
            root,text='Flexor_Tendons:../data/0/FT/0.jpg',
            image = self.Flexor_Tendons_image, 
            compound = 'top'
        )
        self.Flexor_Tendons_label.pack(side='left', anchor=NE)

        self.Carpal_tunnel_image = ImageTk.PhotoImage(
            Image.open('../data/0/CT/0.jpg').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Carpal_tunnel_label = tk.Label(
            root,text='Carpal_tunnel:../data/0/CT/0.jpg',
            image = self.Carpal_tunnel_image, 
            compound = 'top'
        )
        self.Carpal_tunnel_label.pack(side = 'left', anchor=NE)

        # predicted image result
        self.T12_image_predict = ImageTk.PhotoImage(
            Image.open('./icons/placeholder.png').resize(
                (int(width_param * 1.8), int(height_param * 1.8)), Image.ANTIALIAS)
        )
        self.T12_label_predict = tk.Label(
            root, text='Result',anchor ='se',
            image = self.T12_image_predict, 
            compound = 'top',
            fg = "red"
        )
        self.T12_label_predict.place(
            x = int(width_param * 1.5)-100, 
            y = int(height_param * 1.5)+150, 
            width= int(width_param * 1.8), 
            height = int(height_param * 1.8)
        )


        # Median Nerve predicted image configuation
        self.Median_Nerve_image_predict = ImageTk.PhotoImage(
            Image.open('./icons/placeholder.png').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Median_Nerve_label_predict = tk.Label(
            root,text='Median_Nerve predicted:...',anchor ='se',
            image = self.Median_Nerve_image_predict, 
            compound = 'top',
            fg="red"
        )
        self.Median_Nerve_label_predict.place(
            x = int(width_param * 1.5)+420, 
            y = int(height_param * 1.5)+150, 
            width= int(width_param), 
            height = int(height_param)
        )

        # Flexor Tendons predicted image configuation
        self.Flexor_Tendons_image_predict = ImageTk.PhotoImage(
            Image.open('./icons/placeholder.png').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Flexor_Tendons_label_predict = tk.Label(
            root,text='Flexor Tendons predicted:...',anchor ='se',
            image = self.Flexor_Tendons_image_predict, 
            compound = 'top',
            fg="red"
        )
        self.Flexor_Tendons_label_predict.place(
            x = int(width_param * 1.5)+700, 
            y = int(height_param * 1.5)+150, 
            width= int(width_param), 
            height = int(height_param)
        )

        # Carpal tunnel predicted image configuation
        self.Carpal_tunnel_image_predict = ImageTk.PhotoImage(
            Image.open('./icons/placeholder.png').resize(
                (width_param, height_param), Image.ANTIALIAS)
        )
        self.Carpal_tunnel_label_predict = tk.Label(
            root,text='Carpal tunnel predicted:...',anchor ='se',
            image = self.Carpal_tunnel_image_predict, 
            compound = 'top',
            fg="red"
        )
        self.Carpal_tunnel_label_predict.place(
            x = int(width_param * 1.5)+980, 
            y = int(height_param * 1.5)+150, 
            width= int(width_param), 
            height = int(height_param)
        )

    def bind_menu_accelrator_keys(self):
        self.root.bind('<KeyPress-F1>', self.on_about_menu_clicked)
        self.root.bind('<Control-s>', self.on_save_menu_clicked)
        self.root.bind('<Control-S>', self.on_save_menu_clicked)
        self.root.bind('<Control-z>', self.on_undo_menu_clicked)
        self.root.bind('<Control-Z>', self.on_undo_menu_clicked)
        
    @classmethod
    def open_image(cls, filename, width = 224, height = 224):
        img = None
        if not filename:
            return None
        img = ImageTk.PhotoImage(
            Image.open(filename).resize(
                (width, height), Image.ANTIALIAS)
        )
        return img

    @classmethod
    def display_image_on_label(
        cls, 
        label, 
        _image, 
        img_content, 
        text, 
        place_config:tuple = None,
        pack_config:dict = None
    ):
        _image = img_content
        label.configure(image = img_content)
        label.image = img_content
        if place_config:
            label.place(
                x = int(place_config[2] * 1.5)+ place_config[0], 
                y = int(place_config[3] * 1.5)+ place_config[1], 
                width= int(place_config[2]), 
                height = int(place_config[3])
            )
        if pack_config:
            label.pack(**pack_config)
        label.configure(text = text)

    def select_MN_GT_image(self, event = None):
        filename = filedialog.askopenfilename(
            title ='files browser',
            filetypes=[('PNG files', '.png .PNG'), ('.JPG files', '.jpg .JPG'), ('.JPEG files', '.jpeg .JPEG')]
        )
        _image = None   
        if type(filename) is not tuple:
            _image = ImageViewer.open_image(filename)
        if _image is not None:
            ImageViewer.display_image_on_label(
                self.Median_Nerve_label, 
                self.Median_Nerve_image, 
                _image,
                text = "Median_Nerve:" + filename
            )
        self.Median_Nerve_filename = filename
        self.destroy_sequence()
    def select_FT_GT_image(self, event = None):
        
        filename = filedialog.askopenfilename(
            title ='files browser',
            filetypes=[('PNG files', '.png .PNG'), ('.JPG files', '.jpg .JPG'), ('.JPEG files', '.jpeg .JPEG')]
        )
        _image = None   
        if type(filename) is not tuple:
            _image = ImageViewer.open_image(filename)
        if _image is not None:
            ImageViewer.display_image_on_label(
                self.Flexor_Tendons_label, 
                self.Flexor_Tendons_image, 
                _image,
                text = "Flexor_Tendons:" + filename
            )
        self.Flexor_Tendons_filename = filename
        self.destroy_sequence()
    def select_CT_GT_image(self, event = None):
        filename = filedialog.askopenfilename(
            title ='files browser',
            filetypes=[('PNG files', '.png .PNG'), ('.JPG files', '.jpg .JPG'), ('.JPEG files', '.jpeg .JPEG')]
        )
        _image = None   
        if type(filename) is not tuple:
            _image = ImageViewer.open_image(filename)
        if _image is not None:
            ImageViewer.display_image_on_label(
                self.Carpal_tunnel_label, 
                self.Carpal_tunnel_image, 
                _image,
                text = "Carpal_tunnel:" + filename
            )
        self.Carpal_tunnel_filename = filename
        self.destroy_sequence()
    
    def select_T1_image(self, event = None):
        filename = filedialog.askopenfilename(
            title ='files browser',
            filetypes=[('PNG files', '.png .PNG'), ('.JPG files', '.jpg .JPG'), ('.JPEG files', '.jpeg .JPEG')]
        )
        _image = None   
        if type(filename) is not tuple:
            _image = ImageViewer.open_image(filename, width = int(1.5 * 224), height = int(1.5 * 224))
        if _image is not None:
            pack_config =  {'side' : 'left','anchor' : 'nw'}
            ImageViewer.display_image_on_label(
                self.T1_image_label, 
                self.T1_image, 
                _image,
                text = "T1-Image:" + filename,
                pack_config = pack_config
            )
        self.T1_image_filename = filename
        self.destroy_sequence()

    def select_T2_image(self, event = None):
        filename = filedialog.askopenfilename(
            title ='files browser',
            filetypes=[('PNG files', '.png .PNG'), ('.JPG files', '.jpg .JPG'), ('.JPEG files', '.jpeg .JPEG')]
        )
        _image = None   
        if type(filename) is not tuple:
            _image = ImageViewer.open_image(filename, width = int(1.5 * 224), height = int(1.5 * 224))
        if _image is not None:
            pack_config =  {'side' : 'left','anchor' : 'nw'}
            ImageViewer.display_image_on_label(
                self.T2_image_label, 
                self.T2_image, 
                _image,
                text = "T2-Image:" + filename,
                pack_config = pack_config
            )
        self.T2_image_filename = filename
        self.destroy_sequence()

    def display_image_on_all_label(self, filename_list):
        for filename in filename_list:
            if isinstance(filename, tuple):
                filename_ , dc_val = filename
                dc_val_text = "{:.3f}"
                if 'CTOUT' in filename_ :
                    ImageViewer.display_image_on_label(
                        self.Carpal_tunnel_label_predict, 
                        self.Carpal_tunnel_image_predict, 
                        ImageViewer.open_image(filename_),
                        text = "Carpal tunnel predicted:" + filename_ + "\n       CT-DC:" + dc_val_text.format(dc_val),
                        place_config = (980, 150, 225, 225)
                    )
                elif 'FTOUT' in filename_:
                    ImageViewer.display_image_on_label(
                        self.Flexor_Tendons_label_predict, 
                        self.Flexor_Tendons_image_predict, 
                        ImageViewer.open_image(filename_),
                        text = "Flexor Tendons predicted:" + filename_ + "\n        FT-DC:" + dc_val_text.format(dc_val),
                        place_config = (700, 150, 225, 225)
                    )
                elif 'MNOUT' in filename_:
                    ImageViewer.display_image_on_label(
                        self.Median_Nerve_label_predict, 
                        self.Median_Nerve_image_predict, 
                        ImageViewer.open_image(filename_),
                        text = "Median Nerve predicted:" + filename_ + "\n        MN-DC:" + dc_val_text.format(dc_val),
                        place_config = (420, 150, 225, 225)
                    )
                elif 'blended' in filename_:
                    ImageViewer.display_image_on_label(
                        self.T12_label_predict, 
                        self.T12_image_predict, 
                        ImageViewer.open_image(filename_),
                        text = "Blended Result:",
                        place_config = (100, 150, int(225), int(225))
                    )
            else:
                if 'CT' in filename:
                    ImageViewer.display_image_on_label(
                        self.Carpal_tunnel_label, 
                        self.Carpal_tunnel_image, 
                        ImageViewer.open_image(filename),
                        text = "Carpal-Tunnel:" + filename
                    )
                elif 'MN' in filename:
                    ImageViewer.display_image_on_label(
                        self.Median_Nerve_label, 
                        self.Median_Nerve_image, 
                        ImageViewer.open_image(filename),
                        text = "Median-Nerve:" + filename
                    )
                elif 'FT' in filename:
                    ImageViewer.display_image_on_label(
                        self.Flexor_Tendons_label, 
                        self.Flexor_Tendons_image, 
                        ImageViewer.open_image(filename),
                        text = "Flexor-Tendons:" + filename
                    )
                elif 'T1' in filename:
                    ImageViewer.display_image_on_label(
                        self.T1_image_label, 
                        self.T1_image, 
                        ImageViewer.open_image(filename, width = int(1.5 * 224), height = int(1.5 * 224)),
                        text = "T1-Image:" + filename,
                        pack_config = {'side' : 'left','anchor' : 'nw'}
                    )
                elif 'T2' in filename:
                    ImageViewer.display_image_on_label(
                        self.T2_image_label, 
                        self.T2_image, 
                        ImageViewer.open_image(filename, width = int(1.5 * 224), height = int(1.5 * 224)),
                        text = "T2-Image:" + filename,
                        pack_config = {'side' : 'left','anchor' : 'nw'}
                    )
            
    def select_sequence(self, event = None):
        root_directory_path = filedialog.askdirectory()
        if root_directory_path is not None:
            self.Sequence.clear()
            self.Sequence = self.add_sequence_from_directory(root_directory_path = root_directory_path)
            self.display_image_on_all_label(self.Sequence[self.scale_bar_slide_idx])

    def add_sequence_from_directory(self, root_directory_path = None) -> defaultdict(list):

        if not root_directory_path:
            return None
        # get all sub-folder name
        # list [CT,FT,MN,T1,T2]
        image_files_in_directory = defaultdict(list)
        # root_directory_path : /wrist/data/0
        for sub_dirpaths in os.listdir(root_directory_path):
            sub_dirpaths_full = root_directory_path + '/' + sub_dirpaths #eg. /wrist/data/0/T1
            for (dirpath, dirnames, filenames) in os.walk(sub_dirpaths_full):
                for image_file in filenames:
                    if image_file.endswith(".png") or image_file.endswith(".jpg"):
                        # image_idx denote the image number before the Extension, eg. 8.jpg
                        image_idx = int(image_file.split('.', 1 )[0])
                        # image type denote the type of the image usage
                        _image_type = dirpath.rsplit('/', 1)[1]
                        if _image_type == 'CT':
                            _image_type = 0
                        elif _image_type == 'FT':
                            _image_type = 1
                        elif _image_type == 'MN':
                            _image_type = 2
                        elif _image_type == 'T1':
                            _image_type = 3
                        elif _image_type == 'T2':
                            _image_type = 4
                        else:
                            raise Exception('unknown image type {}'.format(_image_type))
                        full_image_path = os.path.join(dirpath + "/" + image_file)
                        '''
                        here we use a dict to store all type of images in one slot, 
                        using the image_idx as the key and the corresponding value would be the list that contain all type of images
                        '''
                        image_files_in_directory[image_idx].insert(_image_type,full_image_path)
        print(image_files_in_directory[0])
        self.destroy_single_selection()
        self.remove_options_from_top_bar()
        return image_files_in_directory

    def get_T1_T2_blended_image(
        self, 
        T1_image_filename, 
        T2_image_filename, 
        save_dir='../eval/',
        ct_countor_tri:tuple = None,
        mn_countor_tri:tuple = None,
        ft_countor_tri:tuple = None, 
    ):
        blended_T1_T2_img = merge_pil_image(T1_image_filename, T2_image_filename,color_space = 'RGB')
        # draw each non-black pixel onto the blended_T1_T2_img
        if ct_countor_tri is not None:
            (ct_countour_filename, pixel_threshold, ct_draw_color) = ct_countor_tri
            ct_countour_img = Image.open(ct_countour_filename)
            for x in range(ct_countour_img.width):
                for y in range(ct_countour_img.height):
                    pixel_val = sum(list(ct_countour_img.getpixel((x,y))))
                    if pixel_val >= pixel_threshold:
                        blended_T1_T2_img.putpixel(
                            (x,y), 
                            ct_draw_color
                        )
        if ft_countor_tri is not None:
            (ft_countour_filename, pixel_threshold, ft_draw_color) = ft_countor_tri
            ft_countour_img = Image.open(ft_countour_filename)
            for x in range(ft_countour_img.width):
                for y in range(ft_countour_img.height):
                    pixel_val = sum(list(ft_countour_img.getpixel((x,y))))
                    if pixel_val >= pixel_threshold:
                        blended_T1_T2_img.putpixel(
                            (x,y), 
                            ft_draw_color
                        )
        if mn_countor_tri is not None:
            (mn_countour_filename, pixel_threshold, mn_draw_color) = mn_countor_tri
            mn_countour_img = Image.open(mn_countour_filename)
            for x in range(mn_countour_img.width):
                for y in range(mn_countour_img.height):
                    pixel_val = sum(list(mn_countour_img.getpixel((x,y))))
                    if pixel_val >= pixel_threshold:
                        blended_T1_T2_img.putpixel(
                            (x,y), 
                            mn_draw_color
                        )
        save_location = save_dir + "blended-" + os.path.basename(T1_image_filename) 
        blended_T1_T2_img.save(save_location)
        return str(save_location)

    def destroy_sequence(self):
        self.Sequence.clear()
        self.scale_bar_slide_idx = 0

    def destroy_single_selection(self):
        self.T1_image_filename = None
        self.T2_image_filename = None
        self.Carpal_tunnel_filename = None
        self.Flexor_Tendons_filename = None
        self.Median_Nerve_filename = None
    def train_options(self):
        print("Train options")

    def eval_options(self):
        """Show evaluation result"""
        Median_nerve_score = 0.0
        Flexor_tendons_score = 0.0
        Carpal_tunnel_score = 0.0
        if not self.Sequence:
            if self.T1_image_filename and self.T2_image_filename:
                # MN
                mn_out_files, mn_countors_outs_files, mn_dc_val_records = predict(
                    input_images = [self.T1_image_filename],
                    target_images = [self.Median_Nerve_filename],
                    config_file="../eval/configMN.yml",
                    save_file_suffix = "MNOUT"
                )
                _image = ImageViewer.open_image(mn_out_files[0])
                ImageViewer.display_image_on_label(
                    self.Median_Nerve_label_predict, 
                    self.Median_Nerve_image_predict, 
                    _image,
                    text = "Flexor Tendons predicted:" + mn_out_files[0],
                    place_config = (420, 150, 225, 225)
                )
                Median_nerve_score = mn_dc_val_records[0]
                # CT
                ct_out_files, ct_countors_outs_files, ct_dc_val_records = predict(
                    input_images = [self.T1_image_filename],
                    target_images = [self.Carpal_tunnel_filename],
                    config_file="../eval/configCT.yml",
                    save_file_suffix = "CTOUT"
                )
                _image = ImageViewer.open_image(ct_out_files[0])
                ImageViewer.display_image_on_label(
                    self.Carpal_tunnel_label_predict, 
                    self.Carpal_tunnel_image_predict, 
                    _image,
                    text = "Carpal tunnel predicted:" + ct_out_files[0],
                    place_config = (980, 150, 225, 225)
                )
                Carpal_tunnel_score = ct_dc_val_records[0]
                # FT
                ft_out_files, ft_countors_outs_files, ft_dc_val_records = predict(
                    input_images = [self.T1_image_filename],
                    target_images = [self.Flexor_Tendons_filename],
                    config_file="../eval/configFT.yml",
                    save_file_suffix = "FTOUT"
                )
                _image = ImageViewer.open_image(ft_out_files[0])
                ImageViewer.display_image_on_label(
                    self.Flexor_Tendons_label_predict, 
                    self.Flexor_Tendons_image_predict, 
                    _image,
                    text = "Flexor Tendons predicted:" + ft_out_files[0],
                    place_config = (700, 150, 225, 225)
                )
                Flexor_tendons_score = ft_dc_val_records[0]

                blended_filename = self.get_T1_T2_blended_image(
                    self.T1_image_filename, 
                    self.T2_image_filename,
                    ct_countor_tri = (ct_countors_outs_files[0], 512, (255,0,0)),
                    ft_countor_tri = (ft_countors_outs_files[0], 512, (0,0,255)),
                    mn_countor_tri = (mn_countors_outs_files[0], 512, (255,255,0))
                )
                    
                ImageViewer.display_image_on_label(
                    label = self.T12_label_predict, 
                    _image = self.T12_image_predict, 
                    img_content = ImageViewer.open_image(blended_filename),
                    text = "Blended Result:",
                    place_config = (100, 150, int(225), int(225))
                )
                    
            eval_result = f"Image DC \n Median nerve : {Median_nerve_score} \nFlexor tendons : {Flexor_tendons_score}  \n Carpal tunnel : {Carpal_tunnel_score}"
        else:
            self.Ouput_Sequence.clear()
            '''
            Output_Sequence layout:
            {
             key:value
             0:(ct_out, ct_out-dc),(mn_out, mn_out-dc),(ft_out, ft_out-dc),(t1t2_out,-1)
            }
            '''
            for idx,(ct, mn, ft, t1, t2) in self.Sequence.items():
                # CT
                ct_out_files, ct_countors_outs_files, ct_dc_val_records = predict(
                    input_images = [t1],
                    target_images = [ct],
                    config_file="../eval/configCT.yml",
                    save_file_suffix = "CTOUT"
                )
                Carpal_tunnel_score = Carpal_tunnel_score + ct_dc_val_records[0]
                self.Ouput_Sequence[idx].insert(0,(ct_out_files[0], ct_dc_val_records[0]))
                # FT
                ft_out_files, ft_countors_outs_files, ft_dc_val_records = predict(
                    input_images = [t1],
                    target_images = [ft],
                    config_file="../eval/configFT.yml",
                    save_file_suffix = "FTOUT"
                )
                Flexor_tendons_score = Flexor_tendons_score + ft_dc_val_records[0]
                self.Ouput_Sequence[idx].insert(1, (ft_out_files[0], ft_dc_val_records[0]))
                # MN
                mn_out_files, mn_countors_outs_files, mn_dc_val_records = predict(
                    input_images = [t1],
                    target_images = [mn],
                    config_file="../eval/configMN.yml",
                    save_file_suffix = "MNOUT"
                )
                Median_nerve_score = Median_nerve_score + mn_dc_val_records[0]
                self.Ouput_Sequence[idx].insert(2, (mn_out_files[0], mn_dc_val_records[0]))

                blended_filename = self.get_T1_T2_blended_image(
                    t1, 
                    t2,
                    ct_countor_tri = (ct_countors_outs_files[0], 512, (255,0,0)),
                    ft_countor_tri = (ft_countors_outs_files[0], 512, (0,0,255)),
                    mn_countor_tri = (mn_countors_outs_files[0], 512, (255,255,0))
                )
                self.Ouput_Sequence[idx].insert(3, (blended_filename, -1))
            Carpal_tunnel_score = Carpal_tunnel_score / (len(self.Sequence))
            Flexor_tendons_score = Flexor_tendons_score / (len(self.Sequence))
            Median_nerve_score = Median_nerve_score / (len(self.Sequence))
            eval_result = f"Sequence DC(mean) \n Median nerve : {Median_nerve_score}\n Flexor tendons : {Flexor_tendons_score} \n Carpal tunnel : {Carpal_tunnel_score}"
        tk.Label(
            self.top_bar,
            text=eval_result
        ).pack(side="left")
        label = tk.Label(self.top_bar)
        label.pack(side="left")

    def function_not_defined(self):
        pass
    
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
    
    def on_scale_bar_clicked(self, event=None):
        if self.Sequence:
            self.scale_bar_slide_idx = int(self.scale_bar.get())
            self.display_image_on_all_label(self.Sequence[self.scale_bar_slide_idx])
            self.display_image_on_all_label(self.Ouput_Sequence[self.scale_bar_slide_idx])

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
