"""
Webpage for tutorial
https://likegeeks.com/python-gui-examples-tkinter-tutorial/
"""

### reaquired packages
# pip install pandastable, tkinter

import os
from NLR import get_recommendation, install_packages
from tkinter import Tk, Label, Button, Entry, filedialog, Radiobutton
from PIL import ImageTk, Image
import tkinter as tk
from pandastable import Table

# debugging
debug = True
pdf_path="/Users/davidkuchelmeister/Google Drive/Projektarbeit 2/Data/Documents/Aleman et al._2005.pdf"
nlr_path="/Users/davidkuchelmeister/Google Drive/Projektarbeit 2/NLR"

# initiate tkinter
window = Tk()

# Window title
window.title("Natural Language Recommender")

# set window size
window.geometry('600x700')

# spacing
y_spacing = 30
x_spacing = 300
level = 0


#=====================
# Add pdf path

path = "./.pictures/title.png"
img = Image.open(path)
img = img.resize((450, 110), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
panel = Label(window, image=img)
panel.image = img
panel.place(x=0, y=y_spacing*level)

#=====================
# Add pdf path

level += 4

# at title
lbl = Label(window, text="Wähle dein PDF-File aus", font=("Arial", 14), fg="grey")
lbl.place(x=0, y=y_spacing*level)

ent1=Entry(window,font=40)
ent1.place(x=x_spacing, y=y_spacing*level)

def browsefunc_pdf():
    global pdf_path
    if not debug: pdf_path=filedialog.askopenfilename()
    ent1.place(x=x_spacing, y=y_spacing*level) # add this

b1=Button(window,text="browse",font=40,command=browsefunc_pdf)
b1.place(x=x_spacing+200, y=y_spacing*level)

#=====================
# Add nlr path

level += 1

# at title
lbl2 = Label(window, text="Wähle den Pfad zu den NLR Dateien", font=("Arial", 14), fg="grey")
lbl2.place(x=0, y=y_spacing*level)

ent2=Entry(window,font=40)
ent2.place(x=x_spacing, y=y_spacing*level)

def browsefunc_nlr():
    global nlr_path
    if not debug: nlr_path=filedialog.askdirectory()
    ent2.place(x=x_spacing, y=y_spacing*level) # add this

b2=Button(window,text="browse",font=40,command=browsefunc_nlr)
b2.place(x=x_spacing+200, y=y_spacing*level)


#=====================
# Schnell / Fortgeschritten

level += 1

# at title
lbl2 = Label(window, text="Algorithmus Eigenschaft:", font=("Arial", 14), fg="grey")
lbl2.place(x=0, y=y_spacing*level)

v = tk.IntVar()
radio_values = {0:"Schnell",1:"Fortgeschritten"}

for i,j in enumerate(radio_values.values()):
    r = Radiobutton(window, 
                text=j,
                padx = 20, 
                variable=v, 
                value=i)
    r.place(x=x_spacing, y=y_spacing*level)
    
    level += 1


#=====================
# Speicher Vergangenheit

# A title
lbl2 = Label(window, text="Speicher Ausführungen:", font=("Arial", 14), fg="grey")
lbl2.place(x=0, y=y_spacing*level)

v2 = tk.IntVar()
radio_values = {0:"Letze",1:"Alle"}

for i,j in enumerate(radio_values.values()):
    r2 = Radiobutton(window, 
                text=j,
                padx = 20, 
                variable=v2, 
                value=i)
    r2.place(x=x_spacing, y=y_spacing*level)
    
    level += 1


#=====================
# Install packages

lbl2 = Label(window, text="Install python packages:", font=("Arial", 14), fg="grey")
lbl2.place(x=0, y=y_spacing*level)

# Add Button Event
def clicked():
    if "nlr_path" in globals():
        install_packages(nlr_path)
    else:
        output['text'] = "Please provide a nlr_path to install packages"

# Add Button
btn = Button(window, text="Install", command=clicked)
btn.place(x=x_spacing, y=y_spacing*level)

# subtitle
level += 1
lbl2 = Label(window, text="(This is only ever needed once on the same Machine)", font=("Arial", 10), fg="grey")
lbl2.place(x=0, y=y_spacing*level-5)

#=====================
# Sumbmit Button

level += 1

# Add Button Event
def clicked():
    os.chdir(nlr_path)
    global predict_df

    loading_text = {0:"1-2",1:"15"}
    output['text'] = f"This process might take {loading_text[v.get()]} minutes"

    predict_df, *_ = get_recommendation(pdf_path=pdf_path, nlr_path=nlr_path, advanced=(v.get()==1), save_past_runs=(v2.get()==1))

    if isinstance(predict_df,str):
        output['text'] = predict_df
    else:
        frame = tk.Frame(window)
        frame.place(x=0, y=y_spacing*level)

        pt = Table(frame)
        pt.model.df = predict_df.sort_values(by="Probability", ascending=False)
        pt.show()

# Add Button
btn = Button(window, text="Submit", command=clicked)
btn.place(x=0, y=y_spacing*level)

#=====================
# Text Output
level += 1
output = Label(window, text="", font=("Arial", 14), fg="grey")
output.place(x=0, y=y_spacing*level)

#=====================
# Run Tkinter
window.mainloop()