import tkinter as tk
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
class ButtonBlock(object):
    def __init__(self, master):
        self.master = master
        
        self.titulo_app = tk.Frame(self.master)
        self.titulo_app.config(bg = 'white')
        self.titulo_app.pack(anchor = 'w', padx = 20)
        titulo = tk.Label(self.titulo_app, text = 'Crea tu propio escenario', fg = 'blue', font = ('Trebuchet MS', 50), bg = 'white')
        titulo.pack()
        
        self.opciones1_frame = tk.Frame(self.master)
        self.opciones1_frame.config(bg = 'white')
        self.opciones1_frame.pack(anchor = 'w', padx = 60, pady = (50, 10))
        
        label_rows = tk.Label(self.opciones1_frame, text = 'Número de filas', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 25))
        label_rows.grid(row=0, column=0, sticky = 'w')
        self.entry_rows = tk.Entry(self.opciones1_frame, font = ('Trebuchet MS', 15))
        self.entry_rows.insert(tk.END, '10')
        self.entry_rows.grid(row=1, column=0, sticky = 'w', padx = 10, pady = 10)        

        self.opciones2_frame = tk.Frame(self.master)
        self.opciones2_frame.config(bg = 'white')
        self.opciones2_frame.pack(anchor = 'w', padx = 60, pady = 10)
        
        label_cols = tk.Label(self.opciones2_frame, text = 'Número de columnas', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 25))
        label_cols.grid(row=0, column=0, sticky = 'w')
        self.entry_cols = tk.Entry(self.opciones2_frame, font = ('Trebuchet MS', 15))
        self.entry_cols.insert(tk.END, '10')
        self.entry_cols.grid(row=1, column=0, sticky = 'w', padx = 10, pady = 10)
        
        self.opciones3_frame = tk.Frame(self.master)
        self.opciones3_frame.config(bg = 'white')
        self.opciones3_frame.pack(anchor = 'w', padx = 60, pady = 60)        
        
        aceptar_button = tk.Button(self.opciones3_frame, text = 'Aceptar', command = self.get_checks,
                                   fg = 'blue', bg = 'white', font = ('Trebuchet MS', 17),
                                   relief = 'groove')
        aceptar_button.grid(row = 0, column = 0, sticky = 'w')
        
        self.button = []
        self.checks = []
        self.checks_values = []
        self.matrix_elements = {}
        
        self.maze_frame = tk.Frame(self.master)
        self.maze_frame.config(bg = 'white')
        self.maze_frame.pack(anchor = 'e')
        self.maze_frame.place(x = 500, y = 150)
    
    def get_checks(self):
        
        self.maze_frame.destroy()
        self.maze_frame = tk.Frame(self.master)
        self.maze_frame.config(bg = 'white', relief = 'groove')
        self.maze_frame.place(x = 500, y = 150)
        
        maze = tk.Frame(self.maze_frame)
        maze.config(bg = 'white', relief = 'groove')
        maze.pack(anchor = 'n')

        opciones_maze = tk.Frame(self.maze_frame)
        opciones_maze.config(bg = 'white', relief = 'groove')
        opciones_maze.pack(anchor = 'w', pady = 15)
        
        self.checks = []
        self.matrix_elements = {}
                
        row_number = int(self.entry_rows.get())
        col_number = int(self.entry_cols.get())
        
        for button in self.button:
            button.destroy()
                
        for row in range(0, row_number):
            row_checks = []
            for col in range(0, col_number):
                
                chk_var = tk.IntVar()
                check = tk.Checkbutton(maze, offvalue = 0, onvalue = 1, variable = chk_var, bg = 'white')
                
                self.matrix_elements[str(row) + str(col)] = chk_var
                row_checks.append(check)
                
                row_checks[-1].grid(row = row, column = col)
                
            self.checks.append(row_checks)
        
        start_label = tk.Label(opciones_maze, text = 'Casilla de comienzo: ', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 10))
        start_label.grid(row = 0, column = 0, sticky = 'w', pady = 5)
        start_entry = tk.Entry(opciones_maze, font = ('Trebuchet MS', 10))
        start_entry.insert(tk.END, "(0,0)")
        
        start_entry.grid(row = 0, column = 1, sticky = 'w', padx = 10)
        start_info_label = tk.Button(opciones_maze, text = 'Info', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 8), height = 1,
                                     command = lambda: tk.messagebox.showinfo(title = 'Casilla de comienzo',
                                                                              message = 'Debes introducir una tupla (fila, columna) no ocupada por obstáculos.\nLa numeración de ambas empieza en 0 en la esquina superior izquierda.'))
        start_info_label.grid(row = 0, column = 2)
        
        exit_label = tk.Label(opciones_maze, text = 'Casilla de llegada: ', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 10))
        exit_label.grid(row = 1, column = 0, sticky = 'w', pady = 5)
        exit_entry = tk.Entry(opciones_maze, font = ('Trebuchet MS', 10))
        exit_entry.insert(tk.END, "({},{})".format(row_number - 1, col_number - 1))
        exit_entry.grid(row = 1, column = 1, sticky = 'w', padx = 10)
        exit_info_label = tk.Button(opciones_maze, text = 'Info', bg = 'blue', fg = 'white', font = ('Trebuchet MS', 8), height = 1,
                                    command = lambda: tk.messagebox.showinfo(title = 'Casilla de llegada',
                                                                              message = 'Debes introducir una tupla (fila, columna) no ocupada por obstáculos.\nLa numeración de ambas empieza en 0 en la esquina superior izquierda.'))
        exit_info_label.grid(row = 1, column = 2)
        
        self.start = list(start_entry.get().replace('(','').replace(')','').split(','))
        self.end = list(exit_entry.get().replace('(','').replace(')','').split(','))  
        self.button.append(tk.Button(self.opciones3_frame, text = 'Obtener escenario', command = self.get_matrix,
                                     fg = 'blue', bg = 'white', font = ('Trebuchet MS', 17),
                                     relief = 'groove'))
        self.button[-1].grid(row = 1, column = 0, pady = 5)    
        
    def get_matrix(self):
        
        self.checks_values = []
        
        row_number = int(self.entry_rows.get())
        col_number = int(self.entry_cols.get())
        
        for row in range(0, row_number):
            row_values = []
            for col in range(0, col_number):
                row_values.append(self.matrix_elements[str(row) + str(col)].get())
            self.checks_values.append(row_values)
            
            
        maze = np.array(self.checks_values)
        
        #print(maze)
        plt.imshow(maze, cmap = 'Greys', interpolation = 'nearest')
        # start_entry
        # 
        
        highlight_cell(int(self.start[0]),int(self.start[1]), color="limegreen", linewidth=3)
        # plt.axis('off')
        # mazeshow = plt.show()
        mazeshow = plt.show()
        # newwin = Toplevel(self.master)
        
        # newwin.title('Escenario')
        # newwin.geometry("200x100") 
        # newwin.resizable(0, 0)
        # canvas = FigureCanvasTkAgg(mazeshow, master=newwin)
        # canvas.show()
        # display = Label(newwin, text="Humm, see a new window !")
        # display.pack()
        os.system('python C:\TFM\Reinforcement-Learning-Maze-master\Reinforcement-Learning-Maze-master\main.py')
        
        
        
        
        return maze
    

    
    

        
        
        
# def combine_funcs(*funcs):
#     def combined_func(*args, **kwargs):
#         for f in funcs:
#             f(*args, **kwargs)
#     return combined_func
def highlight_cell(x,y, ax=None, **kwargs):
        rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect               
        
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1000x600")
    root.config(bg = 'white')
    root.title("Crea tu propio escenario")
    ButtonBlock(root)
    root.mainloop()