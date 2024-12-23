import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from main import *
class EnhancedDualImageGUI:
    def __init__(self, root, process_function):
        self.root = root
        self.root.title("Image Processing GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.process_function = process_function
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.result_text = tk.StringVar()
        
        self.setup_styles()
        self.create_gui()
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure('Main.TFrame', background='#f0f0f0')
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'), background='#f0f0f0')
        style.configure('Custom.TButton', font=('Helvetica', 10), padding=5)
        style.configure('ImageFrame.TLabelframe', background='#ffffff', padding=10)
        style.configure('ImageFrame.TLabelframe.Label', font=('Helvetica', 11, 'bold'))
        
    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="20", style='Main.TFrame')
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        main_frame.grid_columnconfigure(0, weight=1)
        for i in range(4):
            main_frame.grid_rowconfigure(i, weight=1)
            
        title_label = ttk.Label(main_frame, text="Image Processing System", 
                              style='Header.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        input_frame = ttk.Frame(main_frame, style='Main.TFrame')
        input_frame.grid(row=1, column=0, pady=10)
        
        ttk.Label(input_frame, text="Select Input Image:", 
                 style='Header.TLabel').grid(row=0, column=0, padx=5)
        ttk.Entry(input_frame, textvariable=self.input_path, 
                 width=60).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_input, 
                  style='Custom.TButton').grid(row=0, column=2, padx=5)
        
        images_frame = ttk.Frame(main_frame, style='Main.TFrame')
        images_frame.grid(row=2, column=0, pady=20)
        
        input_frame = ttk.LabelFrame(images_frame, text="Input Image", 
                                   style='ImageFrame.TLabelframe')
        input_frame.grid(row=0, column=0, padx=10)
        self.input_preview = ttk.Label(input_frame)
        self.input_preview.grid(padx=10, pady=10)
        
        output_frame = ttk.LabelFrame(images_frame, text="Processed Image", 
                                    style='ImageFrame.TLabelframe')
        output_frame.grid(row=0, column=1, padx=10)
        self.output_preview = ttk.Label(output_frame)
        self.output_preview.grid(padx=10, pady=10)
        
        control_frame = ttk.Frame(main_frame, style='Main.TFrame')
        control_frame.grid(row=3, column=0, pady=20)
        
        process_btn = ttk.Button(control_frame, text="Process Image ▶", 
                               command=self.process_image, style='Custom.TButton')
        process_btn.grid(row=0, column=0, pady=10)
        
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", 
                                     style='ImageFrame.TLabelframe')
        results_frame.grid(row=4, column=0, sticky="ew", pady=(0, 20))
        
        results_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(results_frame, text="Output Path:", 
                 style='Header.TLabel').grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(results_frame, textvariable=self.output_path, 
                 state='readonly', width=80).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Replace Entry with Text widget for better multiline display
        ttk.Label(results_frame, text="Detected Text:", 
                 style='Header.TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.result_text_widget = tk.Text(results_frame, height=6, width=80, wrap=tk.WORD)
        self.result_text_widget.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
    def load_image(self, image_path, preview_label):
        try:
            image = Image.open(image_path)
            image.thumbnail((500, 400))
            photo = ImageTk.PhotoImage(image)
            
            preview_label.configure(image=photo)
            preview_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.PNG *.JPG *.JPEG *.BMP *.TIFF"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.input_path.set(file_path)
            self.load_image(file_path, self.input_preview)
    
    def process_image(self):
        if not self.input_path.get():
            messagebox.showwarning("Warning", "Please select an input image first.")
            return
            
        try:
            output_dir = "data/output"
            os.makedirs(output_dir, exist_ok=True)
            
            processed_image, text = self.process_function(self.input_path.get())
            
            pil_image = Image.fromarray(processed_image)
            
            output_path = os.path.join(output_dir, "output.png")
            pil_image.save(output_path)
            self.output_path.set(output_path)
            
            # Update text widget instead of StringVar
            self.result_text_widget.delete('1.0', tk.END)
            self.result_text_widget.insert('1.0', text)
            
            self.load_image(output_path, self.output_preview)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            
def main():
    root = tk.Tk()
    app = EnhancedDualImageGUI(root, GUI)
    root.mainloop()

if __name__ == "__main__":
    main()