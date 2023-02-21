import datetime
import random
from threading import Thread
from tkinter import *
from tkinter import filedialog, messagebox
from tktooltip import ToolTip

import numpy as np
import pandas as pd

from Ht_thermal_final import thermal_properties, trait_extractor_thermal
from Ht_vs_das import das_properties, trait_extractor_das
from Ht_vs_date import properties, trait_extractor_date
from NDVI_vs_das import NDVI_das_properties, NDVI_trait_extractor_das
from NDVI_vs_date import NDVI_properties, NDVI_trait_extractor_date
from NDVI_vs_thermal import (NDVI_thermal_properties,
                             NDVI_trait_extractor_thermal)


random.seed(20)

Large_Font = ("Verdana", 12, 'bold')
Norm_Font = ("Verdana", 10)
Norm_Font_bold = ("Verdana", 10, 'bold')
Small_font = ("Verdana", 8)
Small_font_bold = ("Verdana", 9, 'bold')

root = Tk()
root.title('Time Series Traits Calculator')

Label(root, text='Select type of data: ', font=Small_font_bold).grid(
    sticky="W", row=0, column=0, padx=10, pady=10)
Label(root, text='Data to import: ', font=Small_font_bold).grid(
    sticky="W", row=2, column=0, padx=10, pady=10)
Label(root, text='Data saved to: ', font=Small_font_bold).grid(
    sticky="W", row=4, column=0, padx=10, pady=10)
Label(text='Fit using: ', font=Small_font_bold).grid(
    row=6, column=0, padx=10, pady=10)
outlier_label = Label(
    root, text='Remove outliers? \n (hover over me to know more)', font=Small_font_bold)
outlier_label.grid(row=11, column=0, padx=10, pady=10)
ToolTip(outlier_label, msg="This option is designed considering error points within drone data. \n If data is clean, it is not required to remove outliers \n But if data is noisy, not removing the outliers may result in \n curve fitting failure because of error datapoints!",
        delay=0, parent_kwargs={"bg": "black"}, fg="#ffffff", bg="#1c1c1c", follow=True)
Label(root, text='Get Specific data corresponding to..:\n (Please choose each attribute once only)',
      font=Small_font).grid(
                            row=14, column=0, padx=10, pady=10)
Label(root, text='Get corresponding Images as well?:', font=Small_font).grid(
                                                                             row=12, column=0, padx=10, pady=10)

list_vals = ['Date at Height(cm)/ NDVI', 'Height/ NDVI at Calender Date(yyyy-mm-dd)', 'Height/ NDVI at Thermal Time',
             'Height/ NDVI at DAS (Days after sowing)', 'Date at %age of max. Height/ NDVI',
             'Date at %age of max. Slope', 'Height/ NDVI at %age of max. Slope', 'Height/ NDVI at %age of max. value']

ndvi_list_vals = ['Sample 1 - 25m altitude S900',
                  'Sample 2 - 12m altitude S900', 'Sample 3 - M600', 'HH - tec 5']


def clear_text(widget):
    widget.grid_forget()


class Cl:

    def __init__(self):
        self.mentioned_perc_maxslp_ht = None
        self.mentioned_perc_maxslp_dt = None
        self.mentioned_perc_maxht = None
        self.mentioned_das = None
        self.mentioned_th_time = None
        self.mentioned_cal_dt = None
        self.mentioned_ht = None
        self.val_at_mentioned_max_perc = None
        self.s_item = None
        self.sh_nm = None
        self.lb = None
        self.img_filename = None
        self.data_entry = None
        self.lb1 = None
        self.output_entry = None
        self.label3 = None
        self.label4 = None
        self.label5 = None
        self.label6 = None
        self.label7 = None
        self.label8 = None
        self.label9 = None
        self.label10 = None
        self.imgLabel = None
        self.ndvilb = None
        self.scrollbar = None
        self.ndviscrollbar = None
        self.entry1 = None
        self.entry2 = None
        self.entry3 = None
        self.entry4 = None
        self.entry5 = None
        self.entry6 = None
        self.entry7 = None
        self.button1 = None
        self.button2 = None
        self.button3 = None
        self.button4 = None
        self.button5 = None
        self.button6 = None
        self.button7 = None
        self.imgBtn = None
        self.sheetselectbtn = None
        self.var5 = IntVar()
        self.var6 = IntVar()
        self.var7 = IntVar()
        self.data_type_var8 = IntVar()
        self.var9 = StringVar(value=list_vals)
        self.ndvi_type_var = StringVar(value=ndvi_list_vals)
        self.varcim = IntVar(value=0)
        self.varoutliers = IntVar(value=0)
        self.output_file = StringVar()
        self.output_file.set('blank')
        self.data_file = StringVar()
        self.data_file.set('blank')
        self.c5 = Checkbutton(root, text='Calender date',
                              variable=self.var5, onvalue=1, offvalue=0)
        self.c5.grid(row=7, column=0, padx=10, pady=10)
        self.c6 = Checkbutton(root, text='Thermal Time', command=self.set_base_temp, variable=self.var6, onvalue=1,
                              offvalue=0)
        self.c6.grid(row=7, column=1, padx=10, pady=10)
        self.c7 = Checkbutton(root, text='DAS (Days after Sowing)',
                              variable=self.var7, onvalue=1, offvalue=0)
        self.c7.grid(row=7, column=2, padx=10, pady=10)
        self.cim = Checkbutton(root, text='Yes', variable=self.varcim, onvalue=1, offvalue=0,
                               command=self.either_show_imgs)
        self.cim.grid(row=12, column=1, padx=10, pady=10, sticky='W')
        self.checkoutliers = Checkbutton(
            root, text='Yes', variable=self.varoutliers, onvalue=1, offvalue=0)
        self.checkoutliers.grid(row=11, column=1, padx=10, pady=10, sticky='W')
        self.r1 = Radiobutton(root, text='Height Data',
                              variable=self.data_type_var8, value=1)
        self.r1.grid(row=0, column=1, padx=5, pady=5)
        self.r2 = Radiobutton(
            root, text='NDVI Data', command=self.select_ndvi_data_type, variable=self.data_type_var8, value=2)
        self.r2.grid(row=0, column=2, padx=5, pady=5)
        self.c8 = None
        self.inputbt = None
        self.var10 = IntVar()
        self.helpButton = None

    def select_ndvi_data_type(self):
        if self.data_type_var8.get() == 2:
            self.ndvilb = Listbox(
                root, listvariable=self.ndvi_type_var, height=2, width=30, selectmode='single')
            self.ndvilb.grid(row=1, column=2, padx=10, pady=10, sticky='W')
            self.ndviscrollbar = Scrollbar(
                root, orient='vertical', command=self.ndvilb.yview)
            self.ndvilb['yscrollcommand'] = self.scrollbar.set
            self.ndviscrollbar.grid(row=1, column=2, sticky='E')

    def either_show_imgs(self):
        if self.varcim.get() == 1:
            self.imgBtn = Button(root, text='Select folder to save images',
                                 command=lambda: Thread(target=self.browse_img_folder).start())
            self.imgBtn.grid(row=12, column=1, padx=10, pady=10, sticky='E')
        else:
            self.imgBtn.grid_forget()

    def browse_img_folder(self):
        self.img_filename = filedialog.askdirectory(
            title='Select folder to save images')
        self.imgLabel = Label(text=self.img_filename)
        self.imgLabel.grid(row=13, column=1, padx=1, pady=1, sticky='E')

    def set_base_temp(self):
        state = self.var6.get()
        if state == 1:
            self.c8 = Checkbutton(root, text='Set Base Temperature (default=0)', command=self.bt_val,
                                  variable=self.var10, onvalue=1, offvalue=0)
            self.c8.grid(row=8, column=1, padx=10, pady=10)
        else:
            self.c8.grid_forget()

    def bt_val(self):
        state1 = self.var10.get()
        if state1 == 1:
            self.inputbt = Text(root, height=1, width=4)
            self.inputbt.grid(sticky="W", row=8, column=2, padx=1, pady=1)

    items_arr = []

    def show_sel_list_items(self):
        self.s_item = self.lb.curselection()
        self.items_arr.append(self.s_item)

        if self.label3:
            if self.label4:
                if self.label5:
                    if self.label6:
                        if self.label7:
                            if self.label8:
                                if self.label9:
                                    for i in self.s_item:
                                        lb_txt = self.lb.get(i)
                                        self.label11 = Label(
                                            text=f'{lb_txt}:', font=Small_font)
                                        self.label10.grid(
                                            row=22, column=1, padx=1, pady=1)
                                        self.entry8 = Entry(
                                            root, justify='center')
                                        self.entry7.grid(
                                            row=22, column=2, padx=1, pady=1)
                                else:
                                    for i in self.s_item:
                                        self.label10 = Label(
                                            text=f'{self.lb.get(i)}:', font=Small_font)
                                        self.label10.grid(
                                            row=21, column=1, padx=1, pady=1)
                                        self.entry7 = Entry(
                                            root, justify='center')
                                        self.entry7.grid(
                                            row=21, column=2, padx=1, pady=1)
                            else:
                                for i in self.s_item:
                                    self.label8 = Label(
                                        text=f'{self.lb.get(i)}:', font=Small_font)
                                    self.label8.grid(
                                        row=20, column=1, padx=1, pady=1)
                                    self.entry6 = Entry(root, justify='center')
                                    self.entry6.grid(
                                        row=20, column=2, padx=1, pady=1)
                        else:
                            for i in self.s_item:
                                self.label7 = Label(
                                    text=f'{self.lb.get(i)}:', font=Small_font)
                                self.label7.grid(
                                    row=19, column=1, padx=1, pady=1)
                                self.entry5 = Entry(root, justify='center')
                                self.entry5.grid(
                                    row=19, column=2, padx=1, pady=1)
                    else:
                        for i in self.s_item:
                            self.label6 = Label(
                                text=f'{self.lb.get(i)}:', font=Small_font)
                            self.label6.grid(row=18, column=1, padx=1, pady=1)
                            self.entry4 = Entry(root, justify='center')
                            self.entry4.grid(row=18, column=2, padx=1, pady=1)
                else:
                    for i in self.s_item:
                        self.label5 = Label(
                            text=f'{self.lb.get(i)}:', font=Small_font)
                        self.label5.grid(row=17, column=1, padx=1, pady=1)
                        self.entry3 = Entry(root, justify='center')
                        self.entry3.grid(row=17, column=2, padx=1, pady=1)
            else:
                for i in self.s_item:
                    self.label4 = Label(
                        text=f'{self.lb.get(i)}:', font=Small_font)
                    self.label4.grid(row=16, column=1, padx=1, pady=1)
                    self.entry2 = Entry(root, justify='center')
                    self.entry2.grid(row=16, column=2, padx=1, pady=1)
        else:
            for i in self.s_item:
                self.label3 = Label(text=f'{self.lb.get(i)}:', font=Small_font)
                self.label3.grid(row=15, column=1, padx=1, pady=1)
                self.entry1 = Entry(root, justify='center')
                self.entry1.grid(row=15, column=2, padx=1, pady=1)

    def choose_specific_data(self):
        self.lb = Listbox(root, listvariable=self.var9,
                          height=3, width=35, selectmode='single')
        self.lb.grid(row=14, column=1, padx=10, pady=10)
        self.scrollbar = Scrollbar(
            root, orient='vertical', command=self.lb.yview)
        self.lb['yscrollcommand'] = self.scrollbar.set
        self.scrollbar.grid(row=14, column=1, sticky='E')

    def show_sel_sheet(self):
        self.sh_nm = self.lb1.get(self.lb1.curselection())
        if self.label9:
            clear_text(self.label9)
        self.label9 = Label(text=f'{self.sh_nm}', font=Small_font)
        self.label9.grid(row=3, column=2, padx=1, pady=1)

    def click_button(self):
        self.button1 = Button(root, text='Open Files', command=lambda: Thread(
            target=self.browse_file).start())
        self.button1.grid(row=2, column=0, padx=10, pady=10, sticky='E')
        self.button2 = Button(root, text='Open Files', command=lambda: Thread(
            target=self.save_file).start())
        self.button2.grid(row=4, column=0, padx=10, pady=10, sticky='E')
        self.button3 = Button(
            root, text='Run', font=Norm_Font_bold, command=self.run_file)
        self.button3.grid(sticky="W", row=25, column=1, padx=20, pady=20)
        self.button5 = Button(
            root, text='Select', font=Small_font, command=self.show_sel_list_items)
        self.button5.grid(row=14, column=2, padx=10, pady=10, sticky='W')
        self.button6 = Button(root, text='Reset',
                              font=Norm_Font, command=self.reset)
        self.button6.grid(sticky="E", row=25, column=1, padx=20, pady=20)
        self.helpButton = Button(
            root, text='Help!', command=self.popup_window, compound='top')
        self.helpButton.grid(row=25, column=0, padx=20, pady=20)
        ToolTip(self.helpButton, msg="Get information about the tool", delay=0,
                parent_kwargs={"bg": "black"}, fg="#ffffff", bg="#1c1c1c", follow=True)

    def popup_window(self):
        window = Toplevel()
        window.title('Tool Information')

        label = Label(window, text='''
        Expected format for Height Data collected from Drone):
            columns: 'Plot', 'ComonVar', 'Nrate', 'Pesticide', 'Sowing date', 'Replicate', 'Row', 'Column', 'Ht_Prcnt99 yyyy-mm-dd'/
            'NDVI_Mean yyyy-mm-dd'

        Expected format for NDVI data collected using different methods):  an excel file should contain aleast these columns: 
            'Plot ID', 'Sowing date', 'Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167', 
            and dates columns with format: 'NDVI_UASInd YYYY-MM-DD sample_1' for Sample 1,
                                            'NDVI_UASInd YYYY-MM-DD sample_2' for Sample 2,
                                            'NDVI_UASInd YYYY-MM-DD sample_3' for Sample 3, and
                                            'NDVI_HHInd YYYY-MM-DD' for tec5 data.

        How to use the tool:
            1. Select the type of input data (height/ NDVI).
            2. Select the file where data is present.
            3. Select the location where you want to save the output file.
                (for option 2 and 3, you can't change location of file from the entry, you need to select it fron the 'Open Files' buttons)
            4. Select atleast one variable to fit data to (Date/ thermal time/ Days after Sowing).
            5. (Optional) Either remove the outliers or not, you will get better idea about how model behaves after 
                and before removing outliers by looking at output images.
            6. (Optional) If you want to get the corrensponding images to the output, select the check button and
                 give the location of folder where you want to save the images.
            7. (Optional) If you want to extract specific traits, select the kind of traits from the list and enter values in the entry boxes.
        ''', justify='left')
        label.pack(fill='x', padx=30, pady=30)

        button_output_info = Button(
            window, text="Understanding Processing of data", command=self.popup_window2, compound='top')
        button_output_info.pack(fill='x')

        button_close = Button(window, text="Close", command=window.destroy)
        button_close.pack(fill='x')

    def popup_window2(self):
        window = Toplevel()
        window.title('Processing Information')

        label = Label(window, text='''
        For Height Data collected by Drone:

            m1/ model1: 
                Two different equations are used. One is Gompertz equation for growth phase and another is Linear equation
                to detect declining phase. The point of intersection of these two equations define the point of max. height (but this is
                true in case only when linear line is having positive slope). When the lines from two equations don't intersect, the
                first point of linear line is used as point of max. height.
            m2/ model2: 
                A single equation is formed by adding the two equations (Gompertz and Linear). Sometimes it find it hard to
                find optimal parameters. But reason of choosing this model is to find consistency that is missing in  model1. Model2 is
                performing better with thermal time data than with height data for existing datasets.
                negative points are replaced with 0.

            Comparison between two models:
                The two models are compared based on r2 score measure. If model2 has higher R2 score, then the traits are recorded from
                model2 as well.

            Output:
                The output consist of an Excel sheet, that will contain all traits data corresponding to the plots in columns. The data
                will be based on the options that the user has chosen.
                Asterisk(*) values are values that may not truly represent the trait.

            Images:
                blue lines represent model 1
                yellow line represent model 2
                green dots are points that are removed (outliers)
                small black star represent intersection point

        For NDVI Data collected by Drone:
            Only plots with SFP tratment are taken.

            Model: A double logistic curve is used for the curve representing NDVI for each row. 
            The model is generally underestimating the height.

            Images:
                Blue line represents the double logistic curve
                Green dots are the points that are removed (outliers)

        Model may not behave for:
            the plots where Nrate is zero,
            the plots where pesticide is RFP,
            the recorded points that are very dodgy and for
            the plots with insufficient data.
        ''', justify='left')
        label.pack(fill='x', padx=20, pady=20)

        button_close = Button(window, text="Close", command=window.destroy)
        button_close.pack(fill='x')

    def browse_file(self):
        filename = filedialog.askopenfilename(title='Open Files', initialdir='/',
                                              filetypes=(("Excel", "*.xlsx"), ("Excel", "*.xls"),
                                                         ("CSV files", "*.csv")))
        self.data_file.set(filename)
        self.data_entry = Entry(root, textvariable=self.data_file, width=50)
        self.data_entry.grid(row=3, column=0, padx=1, pady=1, sticky='E')
        # self.label = Label(text=filename)
        # self.label.grid(row=3, column=0, padx=1, pady=1, sticky='E')
        x1 = pd.ExcelFile(filename)
        self.s_names = x1.sheet_names
        if len(self.s_names) > 1:
            def choose_sheet():
                self.lb1 = Listbox(root, listvariable=StringVar(value=self.s_names), height=2, width=35,
                                   selectmode='single')
                self.lb1.grid(sticky='W', row=2, column=2, padx=1, pady=1)
                self.scrollbar = Scrollbar(
                    root, orient='vertical', command=self.lb1.yview)
                self.lb1['yscrollcommand'] = self.scrollbar.set
                self.scrollbar.grid(row=2, column=2, sticky='E')
                self.button7 = Button(
                    root, text='Select', font=Small_font, command=self.show_sel_sheet)
                self.button7.grid(row=2, column=2, padx=10,
                                  pady=10, sticky='E')

            self.sheetselectbtn = Button(
                root, text='Select Sheet-name', font=Small_font, command=choose_sheet)
            self.sheetselectbtn.grid(row=2, column=1, padx=10, pady=10)

    def save_file(self):
        filename2 = filedialog.asksaveasfilename(title='Open Files', initialdir='/',
                                                 filetypes=(("Excel", "*.xlsx"), ("Excel", "*.xls")))
        if '.xlsx' not in filename2:
            filename2 += '.xlsx'
        self.output_file.set(filename2)
        self.output_entry = Entry(root, textvariable=self.output_file, width=50)
        self.output_entry.grid(row=5, column=0, padx=1, pady=1, sticky='E')
        # self.label2 = Label(text=filename2)
        # self.label2.grid(row=5, column=0, padx=1, pady=1, sticky='E')

    def run_file(self):

        if not self.data_type_var8.get():
            messagebox.showinfo('Error ', 'Please select type of data')
            return

        if self.data_type_var8.get() == 2:
            if self.ndvilb.curselection() == ():
                messagebox.showinfo(
                    'Oops!', 'Seems like you forgot to select the type of NDVI Data')
                return
            else:
                ndvi_method = self.ndvilb.get(self.ndvilb.curselection())
                # if ndvi_method==ndvi_list_val[0]:

        if self.data_file.get() == 'blank':
            messagebox.showinfo(
                'Error ', 'Please select a file to import data from')
            return
        
        if len(self.s_names) > 1:
            if not self.sh_nm:
                messagebox.showinfo('Error ', 'Seems like you forgot to select sheet name!')
                return

        if self.output_file.get() == 'blank':
            messagebox.showinfo(
                'Error ', 'Please select a file to save output data')
            return
        all_check_2 = [self.var5.get(), self.var6.get(), self.var7.get()]   # , self.var7.get()
        y = not np.any(all_check_2)
        if y:
            messagebox.showinfo(
                'Error ', 'Please select at least one variable to fit curve')
            return

        data_input_file = self.data_file.get()
        if self.sh_nm:
            input_sheet_name = self.sh_nm
        data_output_file = self.output_file.get()

        entry_lst = [self.entry1, self.entry2, self.entry3,
                     self.entry4, self.entry5, self.entry6, self.entry7]
        if self.items_arr:
            for i in range(0, len(self.items_arr)):
                if self.items_arr[i][0] == 0:
                    if self.data_type_var8.get() == 1:
                        if self.sh_nm:
                            max_val_ht_1, min_date_1, max_date_1 = properties(
                                data_input_file, sh_name=input_sheet_name)
                        else:
                            max_val_ht_1, min_date_1, max_date_1 = properties(
                                data_input_file)
                        if 0 <= float(entry_lst[i].get()) <= max_val_ht_1:
                            self.mentioned_ht = float(entry_lst[i].get())
                        else:
                            messagebox.showinfo('Error ', f'Mentioned Height is supposed to be in range '
                                                f'0 to {max_val_ht_1}')
                            return
                    elif self.data_type_var8.get() == 2:
                        if self.sh_nm:
                            max_val_ndvi, min_date_2, max_date_2 = NDVI_properties(
                                data_input_file, sh_name=input_sheet_name)
                        else:
                            max_val_ndvi, min_date_2, max_date_2 = NDVI_properties(
                                data_input_file)
                        if 0 <= float(entry_lst[i].get()) <= max_val_ndvi:
                            self.mentioned_ht = float(entry_lst[i].get())
                        else:
                            messagebox.showinfo('Error ', f'Mentioned NDVI value is supposed to be in range '
                                                f'0 to {max_val_ndvi}')
                            return


                if self.items_arr[i][0] == 1:
                    if self.data_type_var8.get() == 1:
                        try:
                            mentioned_dt = datetime.datetime.strptime(
                                str(entry_lst[i].get()), '%Y-%m-%d')
                            if self.sh_nm:
                                max_val_ht_1, min_date_1, max_date_1 = properties(
                                    data_input_file, sh_name=input_sheet_name)
                            else:
                                max_val_ht_1, min_date_1, max_date_1 = properties(
                                    data_input_file)
                            if not min_date_1 <= mentioned_dt <= max_date_1:
                                messagebox.showinfo(
                                    'Error ', f"Mentioned Date should be between {min_date_1} and {max_date_1}")
                                return
                            else:
                                self.mentioned_cal_dt = mentioned_dt.date()
                        except ValueError:
                            messagebox.showinfo(
                                'Error ', "Incorrect mentioned Date format, should be YYYY-MM-DD")
                            return
                    elif self.data_type_var8.get() == 2:
                        try:
                            mentioned_dt = datetime.datetime.strptime(
                                str(entry_lst[i].get()), '%Y-%m-%d')
                            if self.sh_nm:
                                max_val_ndvi, min_date_2, max_date_2 = NDVI_properties(
                                    data_input_file, sh_name=input_sheet_name)
                            else:
                                max_val_ndvi, min_date_2, max_date_2 = NDVI_properties(
                                    data_input_file)
                            if not min_date_2 <= mentioned_dt <= max_date_2:
                                messagebox.showinfo(
                                    'Error ', f"Mentioned Date should be between {min_date_2} and {max_date_2}")
                                return
                            else:
                                self.mentioned_cal_dt = mentioned_dt.date()
                        except ValueError:
                            messagebox.showinfo(
                                'Error ', "Incorrect mentioned Date format, should be YYYY-MM-DD")
                            return
 
                if self.items_arr[i][0] == 2:
                    if self.data_type_var8.get() == 1:
                        if self.sh_nm:
                            mini, maxi = thermal_properties(
                                data_input_file, sh_name=input_sheet_name)
                        else:
                            mini, maxi = thermal_properties(data_input_file)
                        if mini <= float(entry_lst[i].get()) <= maxi:
                            self.mentioned_th_time = entry_lst[i].get()
                        else:
                            messagebox.showinfo('Error ', f'Mentioned Thermal Temperature is supposed to be in range '
                                                f'{mini} to {maxi + 1}')
                            return

                    elif self.data_type_var8.get() == 2:
                        if self.sh_nm:
                            mini, maxi = NDVI_thermal_properties(
                                data_input_file, sh_name=input_sheet_name)
                        else:
                            mini, maxi = NDVI_thermal_properties(
                                data_input_file)
                        if mini <= float(entry_lst[i].get()) <= maxi:
                            self.mentioned_th_time = entry_lst[i].get()
                        else:
                            messagebox.showinfo('Error ', f'Mentioned Thermal Temperature is supposed to be in range '
                                                f'{mini} to {maxi + 1}')
                            return

                if self.items_arr[i][0] == 3:
                    if self.data_type_var8.get() == 1:
                        if self.sh_nm:
                            mini = das_properties(
                                data_input_file, sh_name=input_sheet_name)
                        else:
                            mini = das_properties(data_input_file)
                        if int(entry_lst[i].get()) < mini:
                            messagebox.showinfo(
                                'Error ', f'DAS, first recorded Day is {mini}')
                            return
                        else:
                            self.mentioned_das = entry_lst[i].get()

                    elif self.data_type_var8.get() == 2:
                        if self.sh_nm:
                            mini, maxi = NDVI_das_properties(
                                data_input_file, sh_name=input_sheet_name)
                        else:
                            mini, maxi = NDVI_das_properties(data_input_file)
                        if not (maxi >= int(entry_lst[i].get()) >= mini):
                            messagebox.showinfo(
                                'Error ', f'DAS, first recorded Day is {mini} and last recorded day is {maxi}')
                            return
                        else:
                            self.mentioned_das = entry_lst[i].get()

                if self.items_arr[i][0] == 4:
                    if 0 <= float(entry_lst[i].get()) <= 100:
                        self.mentioned_perc_maxht = entry_lst[i].get()
                    else:
                        messagebox.showinfo(
                            'Error ', 'Mentioned %age value must be numeric(1 to 100)')
                        return

                if self.items_arr[i][0] == 5:
                    if 0 <= float(entry_lst[i].get()) <= 100:
                        self.mentioned_perc_maxslp_dt = entry_lst[i].get()
                    else:
                        messagebox.showinfo(
                            'Error ', 'Mentioned %age value must be numeric(1 to 100)')
                        return

                if self.items_arr[i][0] == 6:
                    if 0 <= float(entry_lst[i].get()) <= 100:
                        self.mentioned_perc_maxslp_ht = entry_lst[i].get()
                    else:
                        messagebox.showinfo(
                            'Error ', 'Mentioned %age value must be numeric(1 to 100)')
                        return

                if self.items_arr[i][0] == 7:
                    if 0 <= float(entry_lst[i].get()) <= 100:
                        self.val_at_mentioned_max_perc = entry_lst[i].get()
                    else:
                        messagebox.showinfo(
                            'Error ', 'Mentioned %age value must be numeric(1 to 100)')
                        return

        dict_ht_date = {}
        dict_ht_thtime = {}
        dict_ht_das = {}

        if self.sh_nm:
            key10, val10 = 'sh_name', self.sh_nm
            dict_ht_date[key10] = val10
            dict_ht_thtime[key10] = val10
            dict_ht_das[key10] = val10
        if self.img_filename:
            key1, val1 = 'img_loc', self.img_filename
            dict_ht_date[key1] = val1
            dict_ht_thtime[key1] = val1
            dict_ht_das[key1] = val1
        if self.mentioned_ht:
            key2, val2 = 'mentioned_ht', self.mentioned_ht
            dict_ht_date[key2] = val2
        if self.mentioned_cal_dt:
            key3, val3 = 'mentioned_cal_dt', self.mentioned_cal_dt
            dict_ht_date[key3] = val3
        if self.mentioned_th_time:
            key4, val4 = 'mentioned_th_time', self.mentioned_th_time
            dict_ht_thtime[key4] = val4
        if self.mentioned_das:
            key5, val5 = 'mentioned_das', self.mentioned_das
            dict_ht_das[key5] = val5
        if self.mentioned_perc_maxht:
            key6, val6 = 'mentioned_perc_maxht', self.mentioned_perc_maxht
            dict_ht_date[key6] = val6
        if self.mentioned_perc_maxslp_dt:
            key7, val7 = 'mentioned_perc_maxslp_dt', self.mentioned_perc_maxslp_dt
            dict_ht_date[key7] = val7
        if self.mentioned_perc_maxslp_ht:
            key8, val8 = 'mentioned_perc_maxslp_ht', self.mentioned_perc_maxslp_ht
            dict_ht_date[key8] = val8
        if self.val_at_mentioned_max_perc:
            key11, val11 = 'val_at_mentioned_max_perc', self.val_at_mentioned_max_perc
            dict_ht_date[key11] = val11
        if self.inputbt:
            key9, val9 = 'base_temp_given', self.inputbt.get("1.0", END)
            dict_ht_thtime[key9] = val9
        if self.varoutliers.get() == 1:
            dict_ht_thtime['outlier_removal'] = 1
            dict_ht_date['outlier_removal'] = 1
            dict_ht_das['outlier_removal'] = 1
        if self.data_type_var8.get() == 2:
            if self.ndvilb.curselection() != ():
                dict_ht_date['ndvi_method'] = ndvi_method
                dict_ht_thtime['ndvi_method'] = ndvi_method
                dict_ht_das['ndvi_method'] = ndvi_method

        messagebox.showinfo(
            "Processing..", "You will be notified when the processing is done!")

        if self.data_type_var8.get() == 1:
            if self.var5.get() == 1 and self.var6.get() == 0 and self.var7.get() == 0:
                output_df = trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df.to_excel(
                    data_output_file, sheet_name='Height_vs_Date')
            elif self.var5.get() == 0 and self.var6.get() == 0 and self.var7.get() == 1:
                output_df = trait_extractor_das(data_input_file, **dict_ht_das)
                output_df.to_excel(
                    data_output_file, sheet_name='Height_vs_Days_after_sowing')
            elif self.var5.get() == 0 and self.var6.get() == 1 and self.var7.get() == 0:
                output_df = trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df.to_excel(
                    data_output_file, sheet_name='Height_vs_thermal_time')
            elif self.var5.get() == 1 and self.var6.get() == 1 and self.var7.get() == 0:
                output_df_dt = trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df_thtime = trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df = pd.merge(output_df_dt, output_df_thtime,
                                     on=['Plot', 'ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column'])
                output_df.to_excel(
                    data_output_file, sheet_name='Height_vs_dt_&_thtime')
            elif self.var5.get() == 0 and self.var6.get() == 1 and self.var7.get() == 1:
                output_df_thtime = trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df_das = trait_extractor_das(
                    data_input_file, **dict_ht_das)
                output_df = pd.merge(output_df_thtime, output_df_das,
                                     on=['Plot', 'ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column'])
                output_df.to_excel(
                    data_output_file, sheet_name='Height_vs_thtime_&_das')
            elif self.var5.get() == 1 and self.var6.get() == 0 and self.var7.get() == 1:
                output_df_dt = trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df_das = trait_extractor_das(
                    data_input_file, **dict_ht_das)
                output_df = pd.merge(output_df_dt, output_df_das,
                                     on=['Plot', 'ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column'])
                output_df.to_excel(
                    data_output_file, sheet_name='Height_vs_dt_&_das')
            elif self.var5.get() == 1 and self.var6.get() == 1 and self.var7.get() == 1:
                output_df_dt = trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df_thtime = trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df_das = trait_extractor_das(
                    data_input_file, **dict_ht_das)
                op_df = pd.merge(output_df_dt, output_df_thtime,
                                 on=['Plot', 'ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column'])
                output_df = pd.merge(op_df, output_df_das,
                                     on=['Plot', 'ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column'])
                output_df.to_excel(
                    data_output_file, sheet_name='Height_vs_all')
            messagebox.showinfo("Processing Complete",
                                "File successfully saved to the given location")

        elif self.data_type_var8.get() == 2:
            if self.var5.get() == 1 and self.var6.get() == 0 and self.var7.get() == 0:
                output_df = NDVI_trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df.to_excel(data_output_file, sheet_name='NDVI_vs_Date')
            elif self.var5.get() == 0 and self.var6.get() == 0 and self.var7.get() == 1:
                output_df = NDVI_trait_extractor_das(
                    data_input_file, **dict_ht_das)
                output_df.to_excel(
                    data_output_file, sheet_name='NDVI_vs_Days_after_sowing')
            elif self.var5.get() == 0 and self.var6.get() == 1 and self.var7.get() == 0:
                output_df = NDVI_trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df.to_excel(
                    data_output_file, sheet_name='NDVI_vs_thermal_time')
            elif self.var5.get() == 1 and self.var6.get() == 1 and self.var7.get() == 0:
                output_df_dt = NDVI_trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df_thtime = NDVI_trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df = pd.merge(output_df_dt, output_df_thtime,
                                     on=['Plot ID', 'Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167'])
                output_df.to_excel(
                    data_output_file, sheet_name='NDVI_vs_dt_&_thtime')
            elif self.var5.get() == 0 and self.var6.get() == 1 and self.var7.get() == 1:
                output_df_thtime = NDVI_trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df_das = NDVI_trait_extractor_das(
                    data_input_file, **dict_ht_das)
                output_df = pd.merge(output_df_thtime, output_df_das,
                                     on=['Plot ID', 'Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167'])
                output_df.to_excel(
                    data_output_file, sheet_name='NDVI_vs_thtime_&_das')
            elif self.var5.get() == 1 and self.var6.get() == 0 and self.var7.get() == 1:
                output_df_dt = NDVI_trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df_das = NDVI_trait_extractor_das(
                    data_input_file, **dict_ht_das)
                output_df = pd.merge(output_df_dt, output_df_das,
                                     on=['Plot ID', 'Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167'])
                output_df.to_excel(
                    data_output_file, sheet_name='NDVI_vs_dt_&_das')
            elif self.var5.get() == 1 and self.var6.get() == 1 and self.var7.get() == 1:
                output_df_dt = NDVI_trait_extractor_date(
                    data_input_file, **dict_ht_date)
                output_df_thtime = NDVI_trait_extractor_thermal(
                    data_input_file, **dict_ht_thtime)
                output_df_das = NDVI_trait_extractor_das(
                    data_input_file, **dict_ht_das)
                op_df = pd.merge(output_df_dt, output_df_thtime,
                                 on=['Plot ID', 'Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167'])
                output_df = pd.merge(op_df, output_df_das,
                                     on=['Plot ID', 'Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167'])
                output_df.to_excel(data_output_file, sheet_name='NDVI_vs_all')
            messagebox.showinfo("Processing Complete",
                                "File successfully saved to the given location")

    def reset(self):
        if self.data_entry:
            clear_text(self.data_entry)
        if self.output_entry:
            clear_text(self.output_entry)
        if self.ndvilb:
            clear_text(self.ndvilb)
        self.var5.set(0)
        self.var6.set(0)
        self.var7.set(0)
        if self.c8:
            clear_text(self.c8)
        if self.inputbt:
            clear_text(self.inputbt)
        if self.var10:
            self.var10.set(0)
            if self.inputbt:
                self.inputbt.delete("1.0", "end")
        if self.label3:
            clear_text(self.label3)
            clear_text(self.entry1)
        if self.label4:
            clear_text(self.label4)
            clear_text(self.entry2)
        if self.label5:
            clear_text(self.label5)
            clear_text(self.entry3)
        if self.label6:
            clear_text(self.label6)
            clear_text(self.entry4)
        if self.label7:
            clear_text(self.label7)
            clear_text(self.entry5)
        if self.label8:
            clear_text(self.label8)
            clear_text(self.entry6)
        if self.label10:
            clear_text(self.label10)
            clear_text(self.entry7)
        if self.items_arr:
            del self.items_arr[:]
        if self.sheetselectbtn:
            clear_text(self.sheetselectbtn)
        if self.lb1:
            clear_text(self.lb1)
        if self.button7:
            clear_text(self.button7)
        if self.label9:
            clear_text(self.label9)
        if self.imgBtn:
            clear_text(self.imgBtn)
        if self.imgLabel:
            clear_text(self.imgLabel)
        self.__init__()
        self.choose_specific_data()


c = Cl()
c.click_button()
c.choose_specific_data()
root.mainloop()
