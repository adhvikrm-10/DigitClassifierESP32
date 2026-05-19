import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import serial
import serial.tools.list_ports

SERIAL_PORT = "COM14"  
BAUD_RATE   = 115200
CELL_SIZE   = 50               
GRID_SIZE   = 8                
SEND_MARKER = 0xFF             
CORRECT_CMD = 0xFE             


class DigitDrawerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("8×8 Digit Drawer — ESP32 Perceptron Inference")
        self.root.resizable(False, False)

        self.grid_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.last_draw_time = 0.0
        self.last_prediction: int | None = None

        self._build_ui()
        self._draw_grid()

    def _build_ui(self) -> None:
        self.canvas = tk.Canvas(
            self.root,
            width=CELL_SIZE * GRID_SIZE,
            height=CELL_SIZE * GRID_SIZE,
            bg="white",
            cursor="pencil",
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 0))
        self.canvas.bind("<Button-1>", self._paint)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-3>", self._erase)   
        self.canvas.bind("<B3-Motion>", self._erase)

        port_frame = tk.Frame(self.root)
        port_frame.grid(row=1, column=0, columnspan=3, pady=(8, 0))

        tk.Label(port_frame, text="Serial port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=SERIAL_PORT)
        self.port_combo = ttk.Combobox(
            port_frame, textvariable=self.port_var, width=22
        )
        self.port_combo["values"] = self._list_ports()
        self.port_combo.pack(side=tk.LEFT, padx=4)

        refresh_btn = tk.Button(
            port_frame, text="refresh", command=self._refresh_ports, width=2
        )
        refresh_btn.pack(side=tk.LEFT)

        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=8)

        self.send_btn = tk.Button(
            btn_frame,
            text="Send to ESP32",
            command=self._send_to_esp32,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=8,
        )
        self.send_btn.pack(side=tk.LEFT, padx=6)

        self.clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            command=self._clear,
            bg="#f44336",
            fg="white",
            font=("Arial", 11),
            padx=8,
        )
        self.clear_btn.pack(side=tk.LEFT, padx=6)

        self.result_var = tk.StringVar(value="Prediction:  —")
        result_label = tk.Label(
            self.root,
            textvariable=self.result_var,
            font=("Arial", 22, "bold"),
            fg="#1a237e",
        )
        result_label.grid(row=3, column=0, columnspan=3, pady=(0, 4))

        correct_frame = tk.Frame(self.root)
        correct_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10))

        tk.Label(correct_frame, text="Correct label (0–9):").pack(side=tk.LEFT)
        self.correct_var = tk.StringVar()
        correct_entry = tk.Entry(
            correct_frame, textvariable=self.correct_var, width=4
        )
        correct_entry.pack(side=tk.LEFT, padx=4)

        correct_btn = tk.Button(
            correct_frame,
            text="Send Correction",
            command=self._send_correction,
        )
        correct_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready.")
        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 9),
            fg="gray",
        )
        status.grid(row=5, column=0, columnspan=3, pady=(0, 4))

    def _draw_grid(self) -> None:
        self.canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                fill = "#111111" if self.grid_data[r, c] else "#f5f5f5"
                self.canvas.create_rectangle(
                    x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE,
                    fill=fill, outline="#cccccc",
                )

    def _paint(self, event: tk.Event) -> None:
        now = time.time()
        if now - self.last_draw_time < 0.02:
            return
        self.last_draw_time = now
        r, c = event.y // CELL_SIZE, event.x // CELL_SIZE
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            self.grid_data[r, c] = 255
            self._draw_grid()

    def _erase(self, event: tk.Event) -> None:
        r, c = event.y // CELL_SIZE, event.x // CELL_SIZE
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            self.grid_data[r, c] = 0
            self._draw_grid()

    def _clear(self) -> None:
        self.grid_data.fill(0)
        self._draw_grid()
        self.result_var.set("Prediction:  —")
        self.last_prediction = None
        self.status_var.set("Grid cleared.")

    @staticmethod
    def _list_ports() -> list[str]:
        return [p.device for p in serial.tools.list_ports.comports()]

    def _refresh_ports(self) -> None:
        ports = self._list_ports()
        self.port_combo["values"] = ports
        if ports:
            self.port_var.set(ports[0])

    def _open_serial(self) -> serial.Serial:
        ser = serial.Serial()
        ser.port = self.port_var.get()
        ser.baudrate = BAUD_RATE
        ser.timeout = 3
        ser.dtr = False        
        ser.rts = False
        ser.open()
        return ser

    def _send_to_esp32(self) -> None:
        flat = self.grid_data.flatten().astype(np.uint8).tobytes()  

        self.status_var.set("Connecting to ESP32...")
        self.root.update()

        try:
            with self._open_serial() as ser:
                time.sleep(0.05)
                ser.reset_input_buffer()   
                ser.write(bytes([SEND_MARKER]))  
                ser.write(flat)                  

                response = ser.readline().decode(errors="ignore").strip()
                if response:
                    try:
                        digit = int(response)
                        self.last_prediction = digit
                        self.result_var.set(f"Prediction:  {digit}")
                        self.status_var.set(
                            f"ESP32 replied: '{response}'"
                        )
                    except ValueError:
                        self.result_var.set(f"ESP32: {response}")
                        self.status_var.set("Unexpected response from ESP32.")
                else:
                    self.result_var.set("Prediction:  [No response]")
                    self.status_var.set(
                        "No reply received. Check USB cable and baud rate."
                    )

        except serial.SerialException as e:
            messagebox.showerror(
                "Serial Error",
                f"Could not open {self.port_var.get()}.\n\n{e}\n\n"
                "Check:\n"
                "  ESP32 is connected via USB\n"
                "  Correct port selected above\n"
                "  No other program is using the port\n"
                "  On Linux: sudo usermod -aG dialout $USER",
            )
            self.status_var.set("Serial error — see dialog.")

    def _send_correction(self) -> None:
        raw = self.correct_var.get().strip()
        if not raw.isdigit() or not (0 <= int(raw) <= 9):
            messagebox.showwarning("Invalid input", "Enter a digit 0–9.")
            return
        label = int(raw)

        try:
            with self._open_serial() as ser:
                time.sleep(0.05)
                ser.reset_input_buffer()   
                ser.write(bytes([CORRECT_CMD, label]))
                response = ser.readline().decode(errors="ignore").strip()
                self.status_var.set(
                    f"Correction sent: {label}  |  ESP32: {response or '(no reply)'}"
                )
        except serial.SerialException as e:
            messagebox.showerror("Serial Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawerApp(root)
    root.mainloop()
