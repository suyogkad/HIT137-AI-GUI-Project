# gui/app_gui.py
# this is the main gui file for our project
# here we join both models (text and image) and show outputs in one app

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# import our own model functions
from models.text_model import analyze_sentiment
from models.image_model import classify_image
from gui.oop_explanation import get_oop_explanation

APP_TITLE = "HIT137 – AI Demo (Text + Image)"
PADDING = 10

# dropdown choices for tasks
CHOICES = [
    "Text (Sentiment)",   
    "Image (Classifier)",  
]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # set window title and size
        self.title(APP_TITLE)
        self.geometry("960x640")
        self.minsize(900, 600)

        # keep track of image path if user selects one
        self.image_path: str | None = None

        # top bar (task dropdown, run button, open image button)
        top = ttk.Frame(self, padding=PADDING)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Task:").pack(side=tk.LEFT)
        self.task = tk.StringVar(value=CHOICES[0])
        ttk.Combobox(top, textvariable=self.task, values=CHOICES, width=24, state="readonly")\
            .pack(side=tk.LEFT, padx=(6, 12))

        self.run_btn = ttk.Button(top, text="Run", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT)

        self.open_btn = ttk.Button(top, text="Open Image…", command=self.on_open)
        self.open_btn.pack(side=tk.LEFT, padx=(6, 0))

        # enable/disable "Open Image" text depending on dropdown selection
        def on_task_change(*_):
            self.open_btn.config(state=tk.NORMAL if "Image" in self.task.get() else tk.DISABLED)
        self.task.trace_add("write", on_task_change)
        on_task_change()

        # main body split into two panes
        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=(0, PADDING))

        left = ttk.Frame(body, padding=PADDING)   
        right = ttk.Frame(body, padding=PADDING)  
        body.add(left, weight=3)
        body.add(right, weight=2)

        # left side (input/output)
        ttk.Label(left, text="Input (type text for sentiment; choose image for classifier):")\
            .pack(anchor="w", pady=(0, 4))

        self.text_input = tk.Text(left, height=6, wrap=tk.WORD)
        self.text_input.pack(fill=tk.X)

        ttk.Label(left, text="Output:").pack(anchor="w", pady=(PADDING, 4))
        self.output = tk.Text(left, height=16, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

        # set font to monospace for better readability
        for widget in (self.text_input, self.output):
            widget.configure(font=("Menlo", 12))

        # right side (info tabs: models + oop explanation)
        info_nb = ttk.Notebook(right)
        info_nb.pack(fill=tk.BOTH, expand=True)

        # tab 1 = model info
        model_info_text = (
            "models used:\n"
            " • text (sentiment): siebert/sentiment-roberta-large-english\n"
            " • image (classifier): google/vit-base-patch16-224\n\n"
            "why these?\n"
            " • both are free, small enough to run locally\n"
            " • they are from different categories (text-classification and image-classification)\n"
            "   which matches assignment requirements\n"
        )
        self.model_info = tk.Text(info_nb, wrap=tk.WORD)
        self.model_info.insert("1.0", model_info_text)
        self.model_info.configure(state=tk.DISABLED, font=("Menlo", 12))
        info_nb.add(self.model_info, text="Model Info")

        # tab 2 = oop explanation
        self.oop_text = tk.Text(info_nb, wrap=tk.WORD)
        self.oop_text.insert("1.0", get_oop_explanation())
        self.oop_text.configure(state=tk.DISABLED, font=("Menlo", 12))
        info_nb.add(self.oop_text, text="OOP Explanation")

    # open file dialog for image
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")],
            initialdir=str(Path.cwd() / "assets"),
        )
        if path:
            self.image_path = path
            messagebox.showinfo("Image selected", f"Using:\n{path}")

    # run button logic with loading indicator
    def on_run(self):
        # first clear output and show loading
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "⏳ loading...\n")
        self.run_btn.config(state=tk.DISABLED)

        task = self.task.get()

        # run in a separate thread so gui doesn’t freeze
        def task_runner():
            try:
                if task.startswith("Text"):
                    text = self.text_input.get("1.0", tk.END).strip()
                    res = analyze_sentiment(text)

                    if res and isinstance(res, list) and "label" in res[0]:
                        label = res[0]["label"]
                        score = res[0]["score"]
                        result_text = (
                            f'text: "{text or "[empty]"}"\n'
                            f"prediction: {label} ({score*100:.2f}%)\n"
                        )
                    else:
                        result_text = f"{res}\n"

                else:  # image classification
                    if not self.image_path:
                        result_text = "please choose an image with “Open Image…”.\n"
                    else:
                        res = classify_image(self.image_path, top_k=5)
                        lines = ["top-5 predictions:"]
                        for i, r in enumerate(res, 1):
                            if "label" in r:
                                lines.append(f"{i}. {r['label']} – {r['score']*100:.2f}%")
                            else:
                                lines.append(f"{i}. {r}")
                        result_text = "\n".join(lines) + "\n"

            except Exception as e:
                result_text = f"⚠️ error: {e}\n"

            # send result back to main gui thread
            self.output.after(0, lambda: self._show_result(result_text))

        threading.Thread(target=task_runner, daemon=True).start()

    # helper function to update output 
    def _show_result(self, text: str):
        self.output.delete("1.0", tk.END)  # remove loading after getting result"
        self.output.insert(tk.END, text)
        self.run_btn.config(state=tk.NORMAL)


def run():
    App().mainloop()