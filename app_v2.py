# -*- coding: utf-8 -*-


import tkinter as tk
import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib

from matplotlib.figure import Figure
import sys
from tkinter import *
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from PIL import Image, ImageTk

import tkinter as tk                # python 3
from tkinter import font as tkfont  # python 3


class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Elige el tipo de input que quieres resolver", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="Matriz manual",
                            command=lambda: controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="Imagen de laberinto",
                            command=lambda: controller.show_frame("PageTwo"))
        button1.pack()
        button2.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Introduce los parámetros de la matriz", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        self.opciones1_frame = tk.Frame(self)
        self.opciones1_frame.config(bg = 'white')
        self.opciones1_frame.pack(anchor = 'w', padx = 60, pady = (50, 10))
        
        label_rows = tk.Label(self.opciones1_frame, text = 'Número de filas', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 25))
        label_rows.grid(row=0, column=0, sticky = 'w')
        self.entry_rows = tk.Entry(self.opciones1_frame, font = ('Trebuchet MS', 15))
        self.entry_rows.insert(tk.END, '10')
        self.entry_rows.grid(row=1, column=0, sticky = 'w', padx = 10, pady = 10)        

        '''Casilla para marcar el número de columnas (longitud del mapa)'''
        self.opciones2_frame = tk.Frame(self)
        self.opciones2_frame.config(bg = 'white')
        self.opciones2_frame.pack(anchor = 'w', padx = 60, pady = 10)
        
        label_cols = tk.Label(self.opciones2_frame, text = 'Número de columnas', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 25))
        label_cols.grid(row=0, column=0, sticky = 'w')
        self.entry_cols = tk.Entry(self.opciones2_frame, font = ('Trebuchet MS', 15))
        self.entry_cols.insert(tk.END, '10')
        self.entry_cols.grid(row=1, column=0, sticky = 'w', padx = 10, pady = 10)
        
        '''Botón de obtener cuadrícula'''
        self.opciones3_frame = tk.Frame(self)
        self.opciones3_frame.config(bg = 'white')
        self.opciones3_frame.pack(anchor = 'w', padx = 60, pady = 60)        
        
        aceptar_button = tk.Button(self.opciones3_frame, text = 'Obtener cuadrícula', command = self.get_checks,
                                   fg = 'blue', bg = 'white', font = ('Trebuchet MS', 17),
                                   relief = 'groove')
        aceptar_button.grid(row = 0, column = 0, sticky = 'w')
        
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()

        '''Inicialización de otras variables'''
        self.button = []
        
        self.maze_frame = tk.Frame(self)
        self.maze_frame.config(bg = 'white')
        self.maze_frame.pack(anchor = 'e')
        self.maze_frame.place(x = 500, y = 150)
        # self.master = master
        
    def get_checks(self):
        
        '''Se elimina el mapa anterior y el botón y se inicializan variables'''        
        self.maze_frame.destroy()
        self.maze_frame = tk.Frame(self)
        self.maze_frame.config(bg = 'white', relief = 'groove')
        self.maze_frame.place(x = 600, y = 150)
        
        for button in self.button:
            button.destroy()        

        self.checks = []
        self.matrix_elements = {}
        
        '''Se crea un nuevo frame con la cuadrícula'''
        maze = tk.Frame(self.maze_frame)
        maze.config(bg = 'white', relief = 'groove')
        maze.pack(anchor = 'n')        
        
        '''Se obtiene el número de filas y columnas de los Entry respectivos'''
        self.row_number = int(self.entry_rows.get())
        self.col_number = int(self.entry_cols.get())
        
        matrix_checks = []        
        for row in range(0, self.row_number):
            row_checks = []
            for col in range(0, self.col_number):
                
                '''Las celdas de la cuadrícula son entradas de texto'''
                check = tk.Entry(maze, bg = 'white', justify = 'center', width = 3)
                
                '''El contenido del Entry correspondiente se guarda en un diccionario'''
                self.matrix_elements[str(row) + str(col)] = check
                row_checks.append(check)
                
                '''El Entry se guarda en una cuadrícula dentro del frame'''
                row_checks[-1].grid(row = row, column = col)
            matrix_checks.append(row_checks)
                
            self.checks.append(row_checks)

        '''Frame para marcar casilla de salida y llegada'''
        opciones_maze = tk.Frame(self.maze_frame)
        opciones_maze.config(bg = 'white', relief = 'groove')
        opciones_maze.pack(anchor = 'w', pady = 15)
        
        start_label = tk.Label(opciones_maze, text = 'Casilla de comienzo: ', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 10))
        start_label.grid(row = 0, column = 0, sticky = 'w', pady = 5)        
        self.start_cell = tk.StringVar()
        start_entry = tk.Entry(opciones_maze, font = ('Trebuchet MS', 10), textvariable = self.start_cell)
        start_entry.insert(tk.END, "(0,0)")
        start_entry.grid(row = 0, column = 1, sticky = 'w', padx = 10)
        start_info_label = tk.Button(opciones_maze, text = 'Info', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 8), height = 1,
                                     command = lambda: tk.messagebox.showinfo(title = 'Casilla de comienzo',
                                                                              message = 'Debes introducir una tupla (fila, columna) no ocupada por obstáculos.\nLa numeración de ambas empieza en 0 en la esquina superior izquierda.'))
        start_info_label.grid(row = 0, column = 2)
        
        exit_label = tk.Label(opciones_maze, text = 'Casilla de llegada: ', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 10))
        exit_label.grid(row = 1, column = 0, sticky = 'w', pady = 5)
        self.exit_cell = tk.StringVar()
        exit_entry = tk.Entry(opciones_maze, font = ('Trebuchet MS', 10), textvariable = self.exit_cell)
        exit_entry.insert(tk.END, "({},{})".format(self.row_number - 1, self.col_number - 1))
        exit_entry.grid(row = 1, column = 1, sticky = 'w', padx = 10)
        exit_info_label = tk.Button(opciones_maze, text = 'Info', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 8), height = 1,
                                    command = lambda: tk.messagebox.showinfo(title = 'Casilla de llegada',
                                                                              message = 'Debes introducir una tupla (fila, columna) no ocupada por obstáculos.\nLa numeración de ambas empieza en 0 en la esquina superior izquierda.'))
        exit_info_label.grid(row = 1, column = 2)
        
        
        def mapaAleatorio():
            
            m=np.random.rand(self.row_number, self.col_number)
            empty_cells=[]
            for row in range(0, m.shape[0]):
                for col in range(0, m.shape[1]):
                    matrix_checks[row][col].delete(0, tk.END)
                    random_number = np.random.rand()
                    if random_number < 0.4:
                        m[row][col] = 0
                        empty_cells.append((row, col))
                    elif random_number <0.6:
                        m[row][col] = 1
                        matrix_checks[row][col].insert(tk.END, str(m[row][col]))
                    else:
                        m[row][col] = round(m[row][col],1)
                        matrix_checks[row][col].insert(tk.END, str(m[row][col]))
            
            random_start_cell = random.choice(empty_cells)
            start_entry.delete(0, tk.END)
            start_entry.insert(tk.END, str(random_start_cell))
            empty_cells.remove(random_start_cell)
            
            exit_entry.delete(0, tk.END)
            exit_entry.insert(tk.END, str(random.choice(empty_cells)))                       
            
        
        random_button = tk.Button(opciones_maze, text = 'Mapa aleatorio', command = mapaAleatorio,
                                  fg = 'blue', bg = 'white', font = ('Trebuchet MS', 9),
                                  relief = 'groove')
        random_button.grid(row = 2, column = 0)
        
        '''Botón para obtener matriz con el mapa codificado, su visualización y la opción de resolver el laberinto'''
        self.button.append(tk.Button(self.opciones3_frame, text = 'Obtener escenario', command = self.get_matrix,
                                     fg = 'blue', bg = 'white', font = ('Trebuchet MS', 17),
                                     relief = 'groove'))
        self.button[-1].grid(row = 1, column = 0, pady = 5)    
    
    '''Método para obtener matriz a partir del contenido de los Entry'''    
    def get_matrix(self):
        
        self.checks_values = []
        
        '''Se rellenan las listas para confeccionar el array posteriormente'''
        for row in range(0, self.row_number):
            row_values = []
            for col in range(0, self.col_number):
                try:
                    row_values.append(round(float(self.matrix_elements[str(row) + str(col)].get()), 1))
                except:
                    row_values.append(0)
            self.checks_values.append(row_values)           
            
        self.matriz = np.array(self.checks_values)
        print(self.matriz)
        
        self.celda_salida = (int(self.start_cell.get().split(',')[0][1:]), int(self.start_cell.get().split(',')[1][:-1]))
        self.celda_llegada = (int(self.exit_cell.get().split(',')[0][1:]), int(self.exit_cell.get().split(',')[1][:-1]))
        
        def highlight_cell(x,y, ax=None, **kwargs):
            rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
            ax = ax or plt.gca()
            ax.add_patch(rect)
            return rect   
        #print(maze)
        plt.imshow(self.matriz, cmap = 'Reds', interpolation = 'nearest')
        # start_entry
        # 
        
        highlight_cell(int(self.celda_salida[1]),int(self.celda_salida[0]), color="limegreen", linewidth=3)
        highlight_cell(int(self.celda_llegada[1]),int(self.celda_llegada[0]), color="blue", linewidth=3)
        
        
       
        return
    

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Resuelve tu imagen", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        
        self.opciones4_frame = tk.Frame(self)
        self.opciones4_frame.config(bg = 'white')
        self.opciones4_frame.pack(anchor = 'w', padx = 60, pady = (50, 10))
        
        
        label_rows = tk.Label(self.opciones4_frame, text = 'Introduce el directorio donde se encuentre tu labreinto', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 25))
        label_rows.grid(row=0, column=0, sticky = 'w')
        self.dire_png = tk.Entry(self.opciones4_frame, font = ('Trebuchet MS', 15))
        # self.dire_png.insert(tk.END, '')
        self.dire_png.grid(row=1, column=0, sticky = 'w', padx = 10, pady = 10)  
        
        self.opciones5_frame = tk.Frame(self)
        self.opciones5_frame.config(bg = 'white')
        self.opciones5_frame.pack(anchor = 'w', padx = 60, pady = 60)        
        
        imprimeLaberinto = tk.Button(self.opciones5_frame, text = 'Obten tu matriz', command = self.png_matrix,
                                   fg = 'blue', bg = 'white', font = ('Trebuchet MS', 17),
                                   relief = 'groove')
        imprimeLaberinto.grid(row = 0, column = 0, sticky = 'w')
        
        
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()
        
     
            
    def png_matrix(self):
        print()
        load = Image.open(str(self.dire_png.get()))
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        
        self.opciones6_frame = tk.Frame(self)
        self.opciones6_frame.config(bg = 'white')
        self.opciones6_frame.pack(anchor = 'w', padx = 60, pady = 60)   
        
        img.place(x=250, y=250)
        im = Image.open(str(self.dire_png.get())).convert('L')
        w, h = im.size
        
        
        binary = im.point(lambda p: p > 128 and 1)
        
       
        binary = binary.resize((w//2,h//2),Image.NEAREST)
        w, h = binary.size
        
        
        nim = np.array(binary)
        
      
        maze = tk.Frame(self)
        maze.config(bg = 'white', relief = 'groove')
        maze.pack(anchor = 'n') 
        
        
        for r in range(h):
           for c in range(w):
               print(nim[r,c],end='')
           print()
        
if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()