import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import nibabel as nib
from matplotlib.colors import ListedColormap
from segmentation_functions import exploration_segmentation

class MedicalImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Viewer - Liver Segmentation")

        # Frame per i controlli a sinistra
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Variabili per i file
        self.image_data = None
        self.label_data = None
        self.current_slice = 0
        self.segmented_mask = None

        # Parametri di segmentazione
        self.tolerance = [80, 140]  # Tolleranza predefinita
        self.filter_size = 4         # Filtro mediano di dimensione predefinita
        self.max_voxel_exploration = 100000  # Limite massimo di voxel esplorabili

        # Inizializza la figura di matplotlib
        self.figure = Figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Pulsanti per caricare file, parametri e segmentare
        self.load_image_button = tk.Button(control_frame, text="Carica Immagine (.nii.gz)", command=self.load_image)
        self.load_image_button.pack()

        self.load_label_button = tk.Button(control_frame, text="Carica Target Label (.nii.gz)", command=self.load_label)
        self.load_label_button.pack()

        self.params_button = tk.Button(control_frame, text="Imposta Parametri", command=self.set_parameters)
        self.params_button.pack()

        self.segment_button = tk.Button(control_frame, text="Esegui Segmentazione", command=self.perform_segmentation)
        self.segment_button.pack()

        # Slider per il contrasto
        self.contrast_slider = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Contrasto", command=self.update_contrast)
        self.contrast_slider.set(128)  # Valore iniziale
        self.contrast_slider.pack()

        # Slider per lo slice
        self.slice_slider = tk.Scale(control_frame, from_=0, to=0, orient=tk.HORIZONTAL, label="Slice", command=self.update_slice)
        self.slice_slider.pack()

        # Checkboxes per controllare la visualizzazione
        self.show_image_var = tk.BooleanVar(value=True)
        self.show_label_var = tk.BooleanVar(value=True)
        self.show_segmented_var = tk.BooleanVar(value=False)

        self.show_image_checkbox = tk.Checkbutton(control_frame, text="Mostra Immagine Originale", variable=self.show_image_var, command=self.update_display)
        self.show_image_checkbox.pack()

        self.show_label_checkbox = tk.Checkbutton(control_frame, text="Mostra Maschera Vera", variable=self.show_label_var, command=self.update_display)
        self.show_label_checkbox.pack()

        self.show_segmented_checkbox = tk.Checkbutton(control_frame, text="Mostra Maschera Calcolata", variable=self.show_segmented_var, command=self.update_display)
        self.show_segmented_checkbox.pack()

        # Frame per le metriche di valutazione
        self.metrics_frame = tk.Frame(control_frame)
        self.metrics_frame.pack()

        # Disabilita i pulsanti inizialmente
        self.segment_button.config(state=tk.DISABLED)

        # Lega la rotellina del mouse per scorrere gli slice
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

    def load_image(self):
        # Carica un file .nii.gz come immagine
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI Files", "*.nii.gz")])
        if file_path:
            nii_image = nib.load(file_path)
            self.image_data = np.array(nii_image.get_fdata(), dtype=np.float32)
            self.current_slice = self.image_data.shape[2] // 2  # Slice centrale
            self.slice_slider.config(to=self.image_data.shape[2] - 1)
            self.slice_slider.set(self.current_slice)
            self.update_display()
            self.check_ready_for_segmentation()

    def load_label(self):
        # Carica un file .nii.gz come target label
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI Files", "*.nii.gz")])
        if file_path:
            nii_label = nib.load(file_path)
            self.label_data = np.array(nii_label.get_fdata(), dtype=np.uint8)
            self.update_display()
            self.check_ready_for_segmentation()

            # Stampa alcune misure della maschera label
            """unique_values, counts = np.unique(self.label_data, return_counts=True)
            print("Valori unici nella maschera label e loro conteggi:")
            for value, count in zip(unique_values, counts):
                print(f"Valore: {value}, Conteggio: {count}")"""

    def update_contrast(self, value):
        self.update_display()

    def update_slice(self, value):
        self.current_slice = int(value)
        self.segmented_mask = None  # Rimuovi la maschera calcolata quando si cambia slice
        self.show_segmented_var.set(False)  # Togli la spunta dalla checkbox della maschera calcolata
        self.update_display()

    def update_display(self):
        # Aggiorna la visualizzazione con immagine e label sovrapposta
        self.ax.clear()
        if self.image_data is not None and self.show_image_var.get():
            image_slice = self.image_data[:, :, self.current_slice]
            contrast_value = int(self.contrast_slider.get())
            image_slice = np.clip(image_slice * (contrast_value / 128), 0, 255)
            self.ax.imshow(image_slice, cmap="gray")
        if self.label_data is not None and self.show_label_var.get():
            label_slice = self.label_data[:, :, self.current_slice]
            mask = np.zeros_like(label_slice, dtype=np.float32)
            mask[label_slice != 0] = 1
            transparent_cmap = ListedColormap(['none', 'red'])
            self.ax.imshow(mask, cmap=transparent_cmap, alpha=0.5, interpolation='none')
        if self.segmented_mask is not None and self.show_segmented_var.get():
            segmented_slice = self.segmented_mask  # Correctly index as 2D array
            transparent_cmap = ListedColormap(['none', 'blue'])
            self.ax.imshow(segmented_slice, cmap=transparent_cmap, alpha=0.5, interpolation='none')
        self.ax.axis("off")
        self.canvas.draw()

    def check_ready_for_segmentation(self):
        # Abilita il pulsante di segmentazione se entrambi i file sono caricati
        if self.image_data is not None and self.label_data is not None:
            self.segment_button.config(state=tk.NORMAL)

    def on_mouse_wheel(self, event):
        # Cambia slice con la rotellina del mouse
        if self.image_data is not None:
            if event.delta > 0:
                self.current_slice = (self.current_slice - 1) % self.image_data.shape[2]
            else:
                self.current_slice = (self.current_slice + 1) % self.image_data.shape[2]
            self.slice_slider.set(self.current_slice)
            self.segmented_mask = None  # Rimuovi la maschera calcolata quando si cambia slice
            self.show_segmented_var.set(False)  # Togli la spunta dalla checkbox della maschera calcolata
            self.update_display()

    def set_parameters(self):
        # Finestra per impostare i parametri
        param_window = Toplevel(self.root)
        param_window.title("Imposta Parametri")

        # Parametri di tolleranza
        tolerance_label = tk.Label(param_window, text="Tolleranza (min, max):")
        tolerance_label.grid(row=0, column=0)
        min_tolerance_entry = tk.Entry(param_window)
        min_tolerance_entry.insert(0, str(self.tolerance[0]))
        min_tolerance_entry.grid(row=0, column=1)
        max_tolerance_entry = tk.Entry(param_window)
        max_tolerance_entry.insert(0, str(self.tolerance[1]))
        max_tolerance_entry.grid(row=0, column=2)

        # Parametri filtro
        filter_label = tk.Label(param_window, text="Dimensione filtro mediano:")
        filter_label.grid(row=1, column=0)
        filter_size_entry = tk.Entry(param_window)
        filter_size_entry.insert(0, str(self.filter_size))
        filter_size_entry.grid(row=1, column=1)

        def update_parameters():
            try:
                self.tolerance = [int(min_tolerance_entry.get()), int(max_tolerance_entry.get())]
                self.filter_size = int(filter_size_entry.get())
                param_window.destroy()
            except ValueError:
                messagebox.showerror("Errore", "Inserisci valori numerici validi.")

        confirm_button = tk.Button(param_window, text="Conferma", command=update_parameters)
        confirm_button.grid(row=2, column=0, columnspan=3)

    def perform_segmentation(self):
        if self.image_data is None or self.label_data is None:
            messagebox.showerror("Errore", "Carica sia l'immagine che il target label prima di segmentare.")
            return

        try:
            slice_data = self.image_data[:, :, self.current_slice]
            label_slice = self.label_data[:, :, self.current_slice]
            self.segmented_mask = exploration_segmentation(slice_data, label_slice, self.tolerance, self.filter_size, self.max_voxel_exploration)
        except ValueError as e:
            messagebox.showerror("Errore", str(e))
            return

        # Aggiorna la visualizzazione per mostrare la maschera calcolata
        self.show_segmented_var.set(True)
        self.update_display()

        # Calcola le misure di accuratezza della segmentazione
        true_positive = np.sum((self.label_data[:,:,self.current_slice] == 1) & (self.segmented_mask == 1))
        false_positive = np.sum((self.label_data[:,:,self.current_slice] == 0) & (self.segmented_mask == 1))
        false_negative = np.sum((self.label_data[:,:,self.current_slice] == 1) & (self.segmented_mask == 0))
        true_negative = np.sum((self.label_data[:,:,self.current_slice] == 0) & (self.segmented_mask == 0))

        dice_coefficient = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
        jaccard_index = true_positive / (true_positive + false_positive + false_negative)
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        f1_score = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

        # Mostra le misure nella finestra del risultato
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        dice_label = tk.Label(self.metrics_frame, text=f"Dice Coefficient: {dice_coefficient:.4f}")
        dice_label.pack()

        jaccard_label = tk.Label(self.metrics_frame, text=f"Jaccard Index: {jaccard_index:.4f}")
        jaccard_label.pack()

        accuracy_label = tk.Label(self.metrics_frame, text=f"Accuracy: {accuracy:.4f}")
        accuracy_label.pack()

        f1_label = tk.Label(self.metrics_frame, text=f"F1 Score: {f1_score:.4f}")
        f1_label.pack()

    def show_result(self, result_mask):
        # Mostra graficamente il risultato della segmentazione
        result_window = Toplevel(self.root)
        result_window.title("Risultato Segmentazione")

        result_figure = Figure(figsize=(6, 6))
        result_ax = result_figure.add_subplot(111)
    
        result_slice = result_mask
        image_slice = self.image_data[:, :, self.current_slice]

        result_ax.imshow(image_slice, cmap="gray")
        result_ax.imshow(result_slice, cmap="Reds", alpha=0.3)
        result_ax.axis("off")

        result_canvas = FigureCanvasTkAgg(result_figure, master=result_window)
        result_canvas.get_tk_widget().pack()
        result_canvas.draw()

# Esegui l'app
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalImageViewerApp(root)
    root.mainloop()
