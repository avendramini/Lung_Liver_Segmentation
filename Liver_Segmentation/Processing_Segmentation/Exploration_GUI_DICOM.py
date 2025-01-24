import os
import pydicom
import numpy as np
import tkinter as tk
from tkinter import messagebox, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.ndimage import median_filter
from collections import deque
from skimage import morphology
from sklearn.metrics import confusion_matrix

# Percorso principale
data_path=".\\Data\\"

# Classe per l'interfaccia grafica
class DicomViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dicom Viewer - Liver Segmentation")

        # Variabili per la navigazione
        self.user_folders = sorted(os.listdir(data_path))
        self.current_user_idx = 0
        self.dicom_files = []
        self.current_image_idx = 0
        self.load_dicom_files()

        # Parametri di default
        self.tolerance = [300, 500]  # Tolleranza predefinita
        self.filter_size = 4         # Filtro mediano di dimensione predefinita
        self.num_samples = 1         # Numero di campioni predefinito

        # Inizializza la figura di matplotlib
        self.figure = Figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Pulsanti per selezione e segmentazione
        self.select_button = tk.Button(self.root, text="Esegui Segmentazione", command=self.perform_segmentation)
        self.select_button.pack()

        self.params_button = tk.Button(self.root, text="Imposta Parametri", command=self.set_parameters)
        self.params_button.pack()

        # Lega la rotellina del mouse alla funzione di navigazione
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

        # Mostra la prima immagine
        self.show_image()

    def load_dicom_files(self):
        # Carica i file DICOM della cartella corrente
        user_folder = self.user_folders[self.current_user_idx]
        user_path = os.path.join(data_path, user_folder)
        max_subfolder = max(os.listdir(user_path), key=int)
        subfolder_path = os.path.join(user_path, max_subfolder)
        images_path = os.path.join(subfolder_path, 'images')
        masks_path = os.path.join(subfolder_path, 'masks')
        
        self.dicom_files = sorted(os.listdir(images_path))
        self.images_path = images_path
        self.masks_path = masks_path

    def show_image(self):
        # Carica e mostra l'immagine corrente con la maschera
        dicom_file = self.dicom_files[self.current_image_idx]
        image_path = os.path.join(self.images_path, dicom_file)
        mask_path = os.path.join(self.masks_path, dicom_file)

        image_data = pydicom.dcmread(image_path)
        image = image_data.pixel_array
        rescale_slope = image_data.get('RescaleSlope', 1)
        rescale_intercept = image_data.get('RescaleIntercept', 0)
        hu_image = image * rescale_slope + rescale_intercept

        mask_data = pydicom.dcmread(mask_path)
        mask = mask_data.pixel_array

        self.ax.clear()
        self.ax.imshow(hu_image, cmap='gray')
        self.ax.imshow(mask, cmap='Reds', alpha=0.3)
        self.ax.axis('off')
        self.canvas.draw()

    def on_mouse_wheel(self, event):
        # Cambia immagine con la rotellina
        if event.delta > 0:
            self.current_image_idx = (self.current_image_idx - 1) % len(self.dicom_files)
        else:
            self.current_image_idx = (self.current_image_idx + 1) % len(self.dicom_files)
        self.show_image()

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

        # Parametro numero campioni
        num_samples_label = tk.Label(param_window, text="Numero di campioni:")
        num_samples_label.grid(row=2, column=0)
        num_samples_entry = tk.Entry(param_window)
        num_samples_entry.insert(0, str(self.num_samples))
        num_samples_entry.grid(row=2, column=1)

        # Bottone per confermare i parametri
        def update_parameters():
            self.tolerance = [int(min_tolerance_entry.get()), int(max_tolerance_entry.get())]
            self.filter_size = int(filter_size_entry.get())
            self.num_samples = int(num_samples_entry.get())
            param_window.destroy()

        confirm_button = tk.Button(param_window, text="Conferma", command=update_parameters)
        confirm_button.grid(row=3, column=0, columnspan=3)

    def perform_segmentation(self):
        # Funzione per eseguire la segmentazione con i parametri impostati
        dicom_file = self.dicom_files[self.current_image_idx]
        image_path = os.path.join(self.images_path, dicom_file)
        mask_path = os.path.join(self.masks_path, dicom_file)

        image_data = pydicom.dcmread(image_path)
        image = image_data.pixel_array
        rescale_slope = image_data.get('RescaleSlope', 1)
        rescale_intercept = image_data.get('RescaleIntercept', 0)
        hu_image = image * rescale_slope + rescale_intercept

        mask_data = pydicom.dcmread(mask_path)
        mask = mask_data.pixel_array

        # Filtro mediano e mascheramento
        filtered_image = median_filter(hu_image, size=self.filter_size)
        mask2 = (filtered_image > self.tolerance[0]) & (filtered_image < self.tolerance[1])

        # Apertura e chiusura morfologiche
        selem = np.ones((3, 3), dtype=np.uint8)
        mask2 = morphology.opening(mask2, selem)
        mask2 = morphology.closing(mask2, selem)

        # Selezione campioni randomici
        coordinates = np.column_stack(np.where((mask == 1) & (mask2 == 1)))
        if len(coordinates) == 0:
            messagebox.showinfo("Info", "Nessun matching trovato, SKIP")
            return

        random_coords = coordinates[np.random.choice(coordinates.shape[0], self.num_samples, replace=False)]
        new_mask = np.zeros(mask.shape)
        vis = np.zeros(mask.shape, dtype=bool)

        # Funzione BFS per propagare la maschera
        def bfs(x, y, tolleranza, vis):
            queue = deque([(x, y)])
            vis[x, y] = True
            new_mask[x, y] = 1
            posx, posy = [0, 0, -1, 1], [-1, 1, 0, 0]
            
            while queue:
                cx, cy = queue.popleft()
                for i in range(4):
                    nx, ny = cx + posx[i], cy + posy[i]
                    if (0 <= nx < new_mask.shape[0] and 0 <= ny < new_mask.shape[1] and
                        tolleranza[0] <= filtered_image[nx, ny] <= tolleranza[1] and
                        not vis[nx, ny] and mask2[nx, ny]):
                        vis[nx, ny] = True
                        new_mask[nx, ny] = 1
                        queue.append((nx, ny))

        # Applica BFS per ogni campione selezionato
        for coord in random_coords:
            bfs(coord[0], coord[1], self.tolerance, vis)
        
        new_mask = (new_mask == 1) & (mask2 == 1)

        # Calcola e stampa metriche
        mask_flat = mask.ravel()
        new_mask_flat = new_mask.ravel()
        conf_matrix = confusion_matrix(mask_flat, new_mask_flat, labels=[0, 1])
        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Visualizzazione del risultato nella nuova finestra
        result_window = Toplevel(self.root)
        result_window.title("Risultato Segmentazione")

        result_fig = Figure(figsize=(6, 6))
        result_ax = result_fig.add_subplot(111)
        result_ax.imshow(hu_image, cmap='gray')
        result_ax.imshow(new_mask, cmap='Reds', alpha=0.3)
        result_ax.axis('off')

        result_canvas = FigureCanvasTkAgg(result_fig, master=result_window)
        result_canvas.get_tk_widget().pack()
        result_canvas.draw()

# Esegui l'app
if __name__ == "__main__":
    root = tk.Tk()
    app = DicomViewerApp(root)
    root.mainloop()
