# gui/app_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from models.text_model import analyze_sentiment
from models.image_model import classify_image
from gui.oop_explanation import get_oop_explanation

APP_TITLE = "HIT137 – AI Demo (Text + Image)"
PADDING = 10

CHOICES = [
    "Text (Sentiment)",     # text -> sentiment label
    "Image (Classifier)",   # image -> ImageNet label
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("960x640")
        self.minsize(900, 600)

        self.image_path = None

        # -------- Top bar --------
        top = ttk.Frame(self, padding=PADDING)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Task:").pack(side=tk.LEFT)
        self.task = tk.StringVar(value=CHOICES[0])
        ttk.Combobox(top, textvariable=self.task, values=CHOICES, width=24, state="readonly")\
            .pack(side=tk.LEFT, padx=(6, 12))

        ttk.Button(top, text="Run", command=self.on_run).pack(side=tk.LEFT)
        self.open_btn = ttk.Button(top, text="Open Image…", command=self.on_open)
        self.open_btn.pack(side=tk.LEFT, padx=(6, 0))

        # enable/disable the Open button based on task
        def on_task_change(*_):
            self.open_btn.config(state=tk.NORMAL if "Image" in self.task.get() else tk.DISABLED)
        self.task.trace_add("write", on_task_change)
        on_task_change()

        # -------- Main panes (left: input/output, right: info) --------
        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=(0, PADDING))

        left = ttk.Frame(body, padding=PADDING)
        right = ttk.Frame(body, padding=PADDING)
        body.add(left, weight=3)
        body.add(right, weight=2)

        # ----- Left column -----
        ttk.Label(left, text="Input (type text for sentiment, choose image for classifier):")\
            .pack(anchor="w", pady=(0, 4))
        self.text_input = tk.Text(left, height=6, wrap=tk.WORD)
        self.text_input.pack(fill=tk.X)
        # no confusing default text
        self.text_input.insert("1.0", "")

        ttk.Label(left, text="Output:").pack(anchor="w", pady=(PADDING, 4))
        self.output = tk.Text(left, height=16, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

        # ----- Right column -----
        info_nb = ttk.Notebook(right)
        info_nb.pack(fill=tk.BOTH, expand=True)

        # Model info tab
        self.model_info = tk.Text(info_nb, wrap=tk.WORD, height=14)
        self.model_info.insert(
            "1.0",
            "Models used:\n"
            " • Text (Sentiment): siebert/sentiment-roberta-large-english\n"
            " • Image (Classifier): google/vit-base-patch16-224\n\n"
            "Why these?\n"
            " • Both are small/free and run locally via Hugging Face Transformers.\n"
            " • They are from different categories (text-classification and image-classification),\n"
            "   which satisfies the assignment requirement."
        )
        self.model_info.configure(state=tk.DISABLED)
        info_nb.add(self.model_info, text="Model Info")

        # OOP explanation tab (from separate module)
        self.oop_text = tk.Text(info_nb, wrap=tk.WORD)
        self.oop_text.insert("1.0", get_oop_explanation())
        self.oop_text.configure(state=tk.DISABLED)
        info_nb.add(self.oop_text, text="OOP Explanation")

        # Nic(er) ttk theme spacing
        for widget in (self.text_input, self.output, self.model_info, self.oop_text):
            widget.configure(font=("Menlo", 12))

    # ---------- actions ----------
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")],
            initialdir=str(Path.cwd() / "assets")
        )
        if path:
            self.image_path = path
            messagebox.showinfo("Image selected", f"Using:\n{path}")

    def on_run(self):
        self.output.delete("1.0", tk.END)
        task = self.task.get()

        if task.startswith("Text"):
            text = self.text_input.get("1.0", tk.END).strip()
            res = analyze_sentiment(text)
            # pretty print sentiment
            if res and isinstance(res, list) and "label" in res[0]:
                label = res[0]["label"]
                score = res[0]["score"]
                self.output.insert(tk.END, f'Text: "{text or "[empty]"}"\n')
                self.output.insert(tk.END, f"Prediction: {label} ({score*100:.2f}%)\n")
            else:
                self.output.insert(tk.END, f"{res}\n")

        else:  # Image (Classifier)
            if not self.image_path:
                self.output.insert(tk.END, "Please choose an image with “Open Image…”.\n")
                return
            res = classify_image(self.image_path, top_k=5)
            # pretty print top-5
            self.output.insert(tk.END, "Top-5 Predictions:\n")
            for i, r in enumerate(res, 1):
                if "label" in r:
                    self.output.insert(tk.END, f"{i}. {r['label']} – {r['score']*100:.2f}%\n")
                else:
                    self.output.insert(tk.END, f"{i}. {r}\n")


def run():
    App().mainloop()
