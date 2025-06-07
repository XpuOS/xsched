import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib
import torch
import time
import torchvision.models as models
import intel_extension_for_pytorch as ipex
import os
import threading

class InferenceWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Model Inference")
        self.set_default_size(600, 400)
        
        # Create a vertical box layout
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(self.box)
        
        # Create a scrolled window for the text view
        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_hexpand(True)
        self.scrolled_window.set_vexpand(True)
        self.box.pack_start(self.scrolled_window, True, True, 0)
        
        # Create a text view for logging
        self.textview = Gtk.TextView()
        self.textview.set_editable(False)
        self.textview.set_wrap_mode(Gtk.WrapMode.WORD)
        self.scrolled_window.add(self.textview)
        
        # Get the text buffer
        self.buffer = self.textview.get_buffer()
        
        # Add a start button
        self.button = Gtk.Button(label="Start Inference")
        self.button.connect("clicked", self.on_start_clicked)
        self.box.pack_start(self.button, False, False, 0)
        
        # Add a quit button
        self.quit_button = Gtk.Button(label="Quit")
        self.quit_button.connect("clicked", Gtk.main_quit)
        self.box.pack_start(self.quit_button, False, False, 0)
        
        # Flag to stop inference
        self.stop_inference = False
    
    def on_start_clicked(self, widget):
        self.button.set_sensitive(False)  # Disable button during inference
        self.stop_inference = False
        self.append_text("Starting inference...\n")
        
        # Start inference in a separate thread
        threading.Thread(target=self.run_inference, daemon=True).start()
    
    def append_text(self, text):
        # This method is thread-safe as it uses GLib.idle_add
        GLib.idle_add(self._do_append_text, text)
    
    def _do_append_text(self, text):
        # Actual text appending (must be called from main thread)
        end_iter = self.buffer.get_end_iter()
        self.buffer.insert(end_iter, text)
        
        # Auto-scroll to the end
        mark = self.buffer.create_mark(None, end_iter, False)
        self.textview.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)
    
    def run_inference(self):
        try:
            self.append_text(f"Process PID: {os.getpid()}\n")
            
            # Load model (this might take some time)
            self.append_text("Loading ResNet152 model...\n")
            model = models.resnet152(weights="ResNet152_Weights.DEFAULT")
            model.eval()
            data = torch.rand(1, 3, 224, 224)
            
            # Move to XPU
            self.append_text("Moving model to XPU...\n")
            model = model.to("xpu")
            data = data.to("xpu")
            model = ipex.optimize(model)
            
            # Warm up
            self.append_text("Warming up (10 iterations)...\n")
            out = 0
            with torch.no_grad():
                for i in range(10):
                    if self.stop_inference:
                        self.append_text("Inference stopped by user\n")
                        return
                    
                    output = model(data)
                    idx = output[0][0].item()
                    out += idx
                    self.append_text(f"Warmup {i+1}/10, output: {idx:.6f}\n")
                    time.sleep(0.1)  # Just to make it visible in UI
            
            # Main inference loop
            self.append_text("Starting main inference for 10 seconds...\n")
            start = time.time()
            cnt = 0
            while time.time() - start < 10 and not self.stop_inference:
                output = model(data)
                idx = output[0][0].item()
                out += idx
                self.append_text(f"Iteration {cnt}, output: {idx:.6f}\n")
                cnt += 1
                time.sleep(0.1)  # Slow down a bit for UI updates
            
            if self.stop_inference:
                self.append_text("Inference stopped by user\n")
            else:
                self.append_text(f"Completed {cnt} iterations in 10 seconds\n")
                self.append_text("Press Quit to exit\n")
            
        except Exception as e:
            self.append_text(f"Error during inference: {str(e)}\n")
        finally:
            GLib.idle_add(self.button.set_sensitive, True)  # Re-enable button

if __name__ == "__main__":
    win = InferenceWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()