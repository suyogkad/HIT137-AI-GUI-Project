import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog

APP_TITLE = "HIT137 — AI GUI (Skeleton) By Sydney Group-28"
APP_SIZE  = "900x650"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_SIZE)
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # main container
        root = ttk.Frame(self, padding=12)
        root.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # header
        header = ttk.Label(root, text="Assignment 3 — Group Sydney 28", font=("Segoe UI", 16, "bold"))
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,10))

        # left side: inputs / outputs
        left = ttk.LabelFrame(root, text="Model I/O", padding=10)
        left.grid(row=1, column=0, sticky="nsew", padx=(0,10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)

        # input controls
        inp_row = ttk.Frame(left)
        inp_row.grid(row=0, column=0, sticky="ew", pady=(0,8))
        inp_row.columnconfigure(1, weight=1)

        ttk.Label(inp_row, text="Image:").grid(row=0, column=0, sticky="w", padx=(0,6))
        self.image_path_var = tk.StringVar(value="")
        ttk.Entry(inp_row, textvariable=self.image_path_var).grid(row=0, column=1, sticky="ew", padx=(0,6))
        ttk.Button(inp_row, text="Browse…", command=self._browse_image).grid(row=0, column=2, sticky="e")

        # buttons placeholder
        btns = ttk.Frame(left)
        btns.grid(row=1, column=0, sticky="ew", pady=(0,8))
        ttk.Button(btns, text="Run Image Classification (placeholder)").grid(row=0, column=0, padx=(0,6))
        ttk.Button(btns, text="Run Image Captioning (placeholder)").grid(row=0, column=1)

        # output area
        ttk.Label(left, text="Output:").grid(row=2, column=0, sticky="w", pady=(6,2))
        self.out = scrolledtext.ScrolledText(left, height=12, wrap="word")
        self.out.grid(row=3, column=0, sticky="nsew")

        # right side: model info + OOP plan
        right = ttk.LabelFrame(root, text="Info / OOP Plan", padding=10)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Selected Models:", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        model_info = (
            "1) Image Classification: 'facebook/deit-tiny-patch16-224' (transformers)\\n"
            "2) Image Captioning: 'Salesforce/blip-image-captioning-base' (transformers)\\n"
            "Note: buttons are placeholders; wiring comes later."
        )
        info = scrolledtext.ScrolledText(right, height=10, wrap="word")
        info.insert("1.0", model_info)
        info.configure(state="disabled")
        info.grid(row=1, column=0, sticky="nsew", pady=(6,0))

        ttk.Label(right, text="OOP Plan:", font=("Segoe UI", 11, "bold")).grid(row=2, column=0, sticky="w", pady=(10,2))
        oop_text = (
            "- Base: ModelRunner(load, run, describe)\\n"
            "- ViTClassifier(ModelRunner)\\n"
            "- BLIPCaptioner(ModelRunner)\\n"
            "- Controller: binds GUI buttons to runners\\n"
            "- View: this Tkinter layout\\n"
        )
        oop = scrolledtext.ScrolledText(right, height=10, wrap="word")
        oop.insert("1.0", oop_text)
        oop.configure(state="disabled")
        oop.grid(row=3, column=0, sticky="nsew")

        # footer
        ttk.Label(root, text="Status: ready (layout only)").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10,0))

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.webp"), ("All files","*.*")]
        )
        if path:
            self.image_path_var.set(path)

if __name__ == "__main__":
    App().mainloop()
