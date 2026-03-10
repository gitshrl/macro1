#!/usr/bin/env python3
"""Macro-AI Desktop — run macro1 tasks on multiple emulators"""

import sys
import os
import subprocess
import threading
import signal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QCheckBox, QComboBox, QPushButton,
    QSpinBox, QGroupBox, QFrame, QScrollArea, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette

DEVICES = {
    "social1 (XL)":   "emulator-5554",
    "social2 (Tsel)": "emulator-5556",
    "social3 (Tsel)": "emulator-5558",
    "social4 (XL)":   "emulator-5560",
}

DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"

running_procs = {}


def get_online_devices():
    try:
        r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        return [l.split("\t")[0] for l in r.stdout.splitlines()[1:] if "\tdevice" in l]
    except:
        return []


class TaskWorker(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, str, str)  # label, serial, output

    def __init__(self, label, serial, prompt, model, max_steps):
        super().__init__()
        self.label = label
        self.serial = serial
        self.prompt = prompt
        self.model = model
        self.max_steps = max_steps

    def run(self):
        self.log_signal.emit(f"🔄 [{self.label}] Starting...\n")
        cmd = [
            sys.executable,
            os.path.expanduser("~/macro1/run_task.py"),
            self.prompt,
            "--device", self.serial,
            "--model", self.model,
            "--max-steps", str(self.max_steps),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.expanduser("~/macro1"),
            )
            running_procs[self.serial] = proc
            output = []
            for line in proc.stdout:
                line = line.rstrip()
                output.append(line)
                self.log_signal.emit(f"[{self.label}] {line}\n")
            proc.wait()
            running_procs.pop(self.serial, None)
            self.done_signal.emit(self.label, self.serial, "\n".join(output))
        except Exception as e:
            running_procs.pop(self.serial, None)
            self.done_signal.emit(self.label, self.serial, f"❌ Error: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Macro-AI")
        self.setMinimumSize(700, 600)
        self.workers = []
        self.done_count = 0
        self.total_count = 0
        self._build_ui()
        self._refresh_devices()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("🤖 Macro-AI")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_layout.addWidget(title)

        # Splitter: top controls + bottom log
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # === TOP: controls ===
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setSpacing(8)

        # Inner splitter: prompt (resizable) + controls below
        inner_splitter = QSplitter(Qt.Orientation.Vertical)
        top_layout.addWidget(inner_splitter)

        # Prompt
        prompt_group = QGroupBox("Instruksi")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("e.g. Open Instagram and like the first 3 posts")
        self.prompt_input.setMinimumHeight(60)
        prompt_layout.addWidget(self.prompt_input)
        inner_splitter.addWidget(prompt_group)

        # Controls below prompt (devices + options + buttons)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        inner_splitter.addWidget(controls_widget)
        inner_splitter.setSizes([120, 300])

        # Devices + Options row
        mid_row = QHBoxLayout()

        # Devices
        dev_group = QGroupBox("Devices")
        dev_layout = QVBoxLayout(dev_group)
        self.device_checks = {}
        self.device_indicators = {}
        for label, serial in DEVICES.items():
            row = QHBoxLayout()
            cb = QCheckBox(label)
            cb.setChecked(True)
            indicator = QLabel("●")
            indicator.setFixedWidth(16)
            row.addWidget(indicator)
            row.addWidget(cb)
            dev_layout.addLayout(row)
            self.device_checks[label] = cb
            self.device_indicators[label] = indicator
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedHeight(28)
        refresh_btn.clicked.connect(self._refresh_devices)
        dev_layout.addWidget(refresh_btn)
        mid_row.addWidget(dev_group)

        # Options
        opt_group = QGroupBox("Options")
        opt_layout = QVBoxLayout(opt_group)
        
        opt_layout.addWidget(QLabel("Max Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(5, 100)
        self.steps_spin.setSingleStep(5)
        self.steps_spin.setValue(20)
        opt_layout.addWidget(self.steps_spin)
        opt_layout.addStretch()
        mid_row.addWidget(opt_group)

        controls_layout.addLayout(mid_row)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; font-weight: bold; border-radius: 6px;")
        self.run_btn.clicked.connect(self._run)
        
        self.stop_btn = QPushButton("⏹  Stop All")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-size: 14px; font-weight: bold; border-radius: 6px;")
        self.stop_btn.clicked.connect(self._stop_all)
        
        self.copy_btn = QPushButton("📋  Copy Log")
        self.copy_btn.setFixedHeight(40)
        self.copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.log_output.toPlainText()))

        self.clear_btn = QPushButton("🗑  Clear Log")
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.clicked.connect(lambda: self.log_output.clear())
        
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.clear_btn)
        controls_layout.addLayout(btn_row)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-size: 12px;")
        controls_layout.addWidget(self.status_label)

        splitter.addWidget(top_widget)

        # === BOTTOM: log output ===
        log_group = QGroupBox("Output Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Monospace", 10))
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; selection-background-color: #264f78; selection-color: #ffffff;")
        log_layout.addWidget(self.log_output)
        splitter.addWidget(log_group)

        splitter.setSizes([320, 280])

    def _refresh_devices(self):
        online = get_online_devices()
        for label, serial in DEVICES.items():
            ind = self.device_indicators[label]
            if serial in online:
                ind.setStyleSheet("color: #4CAF50; font-size: 16px;")
            else:
                ind.setStyleSheet("color: #f44336; font-size: 16px;")
        self.status_label.setText(f"Devices: {len(online)}/4 online")

    def _log(self, text):
        self.log_output.moveCursor(self.log_output.textCursor().MoveOperation.End)
        self.log_output.insertPlainText(text)
        self.log_output.ensureCursorVisible()

    def _run(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            self.status_label.setText("❌ Prompt kosong!")
            return

        selected = [l for l, cb in self.device_checks.items() if cb.isChecked()]
        if not selected:
            self.status_label.setText("❌ Pilih minimal 1 device!")
            return

        online = get_online_devices()
        serials = [(l, DEVICES[l]) for l in selected if DEVICES[l] in online]
        if not serials:
            self.status_label.setText("❌ Semua device offline!")
            return

        model = DEFAULT_MODEL
        max_steps = self.steps_spin.value()

        self.done_count = 0
        self.total_count = len(serials)
        self.workers = []
        self.run_btn.setEnabled(False)
        self.status_label.setText(f"🚀 Running on {len(serials)} device(s)...")
        self._log(f"\n{'='*60}\n🚀 Task: {prompt}\n📱 Devices: {', '.join(l for l,_ in serials)}\n{'='*60}\n\n")

        for label, serial in serials:
            w = TaskWorker(label, serial, prompt, model, max_steps)
            w.log_signal.connect(self._log)
            w.done_signal.connect(self._on_done)
            w.start()
            self.workers.append(w)

    def _on_done(self, label, serial, output):
        self.done_count += 1
        self._log(f"\n✅ [{label}] DONE ({self.done_count}/{self.total_count})\n")
        if self.done_count >= self.total_count:
            self.run_btn.setEnabled(True)
            self.status_label.setText(f"✅ All {self.total_count} tasks completed!")

    def _stop_all(self):
        killed = 0
        for serial, proc in list(running_procs.items()):
            try:
                proc.send_signal(signal.SIGTERM)
                killed += 1
            except:
                pass
        running_procs.clear()
        for w in self.workers:
            w.terminate()
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"🛑 Killed {killed} task(s)")
        self._log(f"\n🛑 Stopped {killed} task(s)\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
