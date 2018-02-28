# Patel, Nabilahmed
# 1001-234-817
# 2016-10-09
# Assignment_04

import numpy as np
import Tkinter as Tk
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys

class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Widrow-Huff learning in 2d space.
    Nabilahmed Patel 2016_10_09
    """

    def __init__(self, master):
        
        #reading data
        self.data = np.loadtxt("stock_data.txt", skiprows=1, delimiter=',', dtype=np.float32)
        #normalizing data
        max_price = np.max(self.data[0:,0])
        max_volume = np.max(self.data[0:,1])
        self.data[:,0] /= max_price
        self.data[:,1] /= max_volume
        
        self.master = master
        #Create Plot Area
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0.0
        self.ymax = 0.5
        self.master.update()
        self.min_initial_weights = -0.1         # minimum initial weight
        self.max_initial_weights = 0.1          # maximum initial weight
        self.number_of_inputs = 2               # number of inputs to the network
        self.learning_rate = 0.1                # learning rate
        self.batch_size = 0                     # 0 := entire trainingset as a batch
        self.number_of_delayed_elements = 0     #delayed elements
        self.number_of_iteration = 5            #number of iteration over whole sample size
        self.number_of_classes = 2
        self.sample_size = 10                   #sample_sizde in percentage
        
        self.weights = np.random.uniform(self.min_initial_weights,self.max_initial_weights,(2,(2+(2*self.number_of_delayed_elements)+1)))    #min_weight,max_weight,structure or dimension        
        
        self.xx = np.array([])
        self.yy = np.array([])

        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=1, uniform="group1")		
        self.master.columnconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(2, weight=1, uniform="group1")
        
        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=3, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        #self.figure = plt.figure("Multiple Linear Classifiers")
        #self.axes = self.figure.add_subplot(111)
        plt.title("Widrow-Huff Learning")
        plt.ylabel('Error')
        plt.xlabel('Batch No.')
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

	# Create sliders frame1
        self.sliders_frame1 = Tk.Frame(self.master)
        self.sliders_frame1.grid(row=1, column=0)
        self.sliders_frame1.rowconfigure(0, weight=1)
        self.sliders_frame1.columnconfigure(0, weight=1, uniform='s1')
        
	# Create sliders frame2
        self.sliders_frame2 = Tk.Frame(self.master)
        self.sliders_frame2.grid(row=1, column=1)
        self.sliders_frame2.rowconfigure(0, weight=1)
        self.sliders_frame2.columnconfigure(0, weight=1, uniform='s1')
		
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=2)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')


        # Set up the sliders1
        self.number_of_iteration_slider_label = Tk.Label(self.sliders_frame1, text="No. Of Iteration")
        self.number_of_iteration_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_iteration_slider = Tk.Scale(self.sliders_frame1, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=1, to_=100, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)        
        self.number_of_iteration_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_iteration_slider_callback())
        self.number_of_iteration_slider.set(self.number_of_iteration)
        self.number_of_iteration_slider.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.number_of_delayed_elements_slider_label = Tk.Label(self.sliders_frame1, text="No. Of Delayed Elements")
        self.number_of_delayed_elements_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_delayed_elements_slider = Tk.Scale(self.sliders_frame1, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=1, to_=100, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)        
        self.number_of_delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_delayed_elements_slider_callback())
        self.number_of_delayed_elements_slider.set(self.number_of_delayed_elements)
        self.number_of_delayed_elements_slider.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # Set up the sliders2
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame2, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame2, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.001, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.batch_size_slider_label = Tk.Label(self.sliders_frame2, text="Batch Size")
        self.batch_size_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame2, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=0, to_=1000, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.sample_size_slider_label = Tk.Label(self.sliders_frame2, text="Samples Size (%)")
        self.sample_size_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_slider = Tk.Scale(self.sliders_frame2, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=1, to_=100, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)        
        self.sample_size_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_slider_callback())
        self.sample_size_slider.set(self.sample_size)
        self.sample_size_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        # Set up the slider and buttons        
        self.set_weight_to_zero_bottun = Tk.Button(self.buttons_frame,
                                                   text="Set Weight to Zero",
                                                   bg="yellow", fg="red",
                                                   command=lambda: self.set_weight_to_zero_bottun_callback())
        self.set_weight_to_zero_bottun.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

    def display_epoch(self):
        self.axes.cla()
        batch_number = np.array([i for i in range(self.price_MSE.shape[0])])
        self.axes.plot(batch_number, self.price_MSE,'r', label='price MSE')
        self.axes.plot(batch_number,self.volume_MSE,'b', label='volume MSE')
        self.axes.plot(batch_number, self.price_MAE,'g',label='price MAE')
        self.axes.plot(batch_number,self.volume_MAE,'y', label='volume MAE')
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title("Widrow-Huff")
        plt.ylabel('Error')
        plt.xlabel('Batch No.')
        plt.legend()
        self.canvas.draw()

    def number_of_iteration_slider_callback(self):
        self.number_of_iteration = self.number_of_iteration_slider.get()

    def number_of_delayed_elements_slider_callback(self):
        self.number_of_delayed_elements = self.number_of_delayed_elements_slider.get()
        self.weights = np.random.uniform(self.min_initial_weights,self.max_initial_weights,(2,(2+(2*self.number_of_delayed_elements)+1)))    #min_weight,max_weight,structure or dimension
        
    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()

    def sample_size_slider_callback(self):
        self.sample_size = self.sample_size_slider.get()

    def set_weight_to_zero_bottun_callback(self):
        temp_text = self.set_weight_to_zero_bottun.config('text')[-1]
        self.set_weight_to_zero_bottun.config(text='Please Wait')
        self.set_weight_to_zero_bottun.update_idletasks()
        self.weights = np.zeros((2,(2+(2*self.number_of_delayed_elements)+1)))
        self.set_weight_to_zero_bottun.config(text=temp_text)
        self.set_weight_to_zero_bottun.update_idletasks()

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        
        batch_size = self.batch_size
        no_of_delayed_elements = self.number_of_delayed_elements
        sample_size = self.sample_size
        alpha = self.learning_rate
        no_of_samples = self.data.shape[0] * sample_size / 100

        input_samples = self.data[0:no_of_samples]
        
        if batch_size == 0:
            no_of_batches = 1
        else:
            no_of_batches = (no_of_samples/batch_size) if (no_of_samples%batch_size) == 0 else (no_of_samples/batch_size) + 1

        self.xmax = no_of_batches
        
        for l in range(self.number_of_iteration):
            
            self.price_MSE = np.array([])
            self.volume_MSE = np.array([])
            self.price_MAE = np.array([])
            self.volume_MAE = np.array([])
            
            for i in range(no_of_batches):

                #for the first batch special case is first no_of_deleyaed_elements elements will not be calculated for error
                if i == 0:
                    start_index = no_of_delayed_elements
                    if batch_size == 0:
                        end_index = no_of_samples
                    else:
                        end_index = batch_size
                else:
                    start_index = j + 1
                    end_index = start_index + batch_size

                for j in range(start_index, end_index):
                    if j == (no_of_samples - 1):
                        break
                    
                    #next sample(element) is target
                    t = np.transpose(input_samples[j+1])    
                    t = t.reshape(-1,1)
                    #creating input_vector P and arranging it
                    P = input_samples[(j-no_of_delayed_elements):(j+1)]
                    P = np.concatenate((P[0:,0],P[0:,1]),axis=0)
                    P = P.reshape(-1,1)
                    #adding Bias input
                    P = np.vstack([P, np.ones((1, P.shape[1]), float)])
                    #calculating an output
                    a = np.dot(self.weights,P)
                    #calculating an error
                    e = t-a
                    #adjusting the weight
                    self.weights = self.weights + np.dot(2*alpha*e,np.transpose(P))
                    
                price_Err = np.array([])
                volume_Err = np.array([])
                Eprice_mae = np.array([])
                Evolume_mae = np.array([])
               
                for k in range(start_index, end_index):
                    if k == (no_of_samples - 1):
                        break
                    
                    #next sample(element) is target
                    T = np.transpose(input_samples[k+1])    
                    T = T.reshape(-1,1)
                    
                    #calculating an output
                    A = np.dot(self.weights,P)
                    
                    #calculating an error
                    E = T-A
                    Eprice_mae = np.append(Eprice_mae,np.absolute(E[0]))
                    Evolume_mae = np.append(Evolume_mae,np.absolute(E[1]))

                    price_Err = np.append(price_Err,np.square(E[0]))
                    volume_Err = np.append(volume_Err,np.square(E[1]))

                #finding max error for each batch
                price_mae = np.max(Eprice_mae)
                volume_mae = np.max(Evolume_mae)

                #finding mean square error for each batch
                price_mse = np.mean(price_Err)
                volume_mse = np.mean(volume_Err)

                self.price_MSE = np.append(self.price_MSE,price_mse)
                self.volume_MSE = np.append(self.volume_MSE,volume_mse)
                self.price_MAE = np.append(self.price_MAE,price_mae)
                self.volume_MAE = np.append(self.volume_MAE,volume_mae)
                self.display_epoch()
                
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

if __name__ == "__main__":
    
    main_frame = Tk.Tk()
    main_frame.title("Widrow-Huff")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame)
    main_frame.mainloop()
