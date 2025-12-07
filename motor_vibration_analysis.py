import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats
from scipy.fft import fft

# --- 解决matplotlib中文乱码问题的代码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# -----------------------------------------

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("电机振动故障预判系统 (AI版)")
        self.master.geometry("1200x800")
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()
        self.generate_and_train()
        self.running = False

    def create_widgets(self):
        # Main layout frames
        control_frame = ttk.LabelFrame(self, text="控制面板")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plot_frame = ttk.LabelFrame(self, text="实时振动信号 & 故障分析")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel widgets
        self.start_button = ttk.Button(control_frame, text="开始实时监测", command=self.start_monitoring)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(control_frame, text="停止监测", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.status_label = ttk.Label(control_frame, text="状态: 未运行", foreground="red")
        self.status_label.pack(pady=10)

        self.results_tree = ttk.Treeview(control_frame, columns=("Time", "Status", "Fault Type", "Probability"), show="headings")
        self.results_tree.heading("Time", text="时间")
        self.results_tree.heading("Status", text="状态")
        self.results_tree.heading("Fault Type", text="故障类型")
        self.results_tree.heading("Probability", text="置信度")
        self.results_tree.column("Time", width=120)
        self.results_tree.column("Status", width=80)
        self.results_tree.column("Fault Type", width=120)
        self.results_tree.column("Probability", width=80)
        self.results_tree.pack(pady=10, fill=tk.Y, expand=True)

        # Plotting area
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_and_train(self):
        # 1. Generate simulated data
        self.fault_types = {0: "正常", 1: "轴承磨损", 2: "转子不平衡", 3: "地脚螺栓松动"}
        X, y = self.generate_fault_data(1000)

        # 2. Feature Extraction
        features = np.array([self.extract_features(signal) for signal in X])

        # 3. Train KMeans for anomaly detection
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        # We assume normal data is the larger cluster
        normal_cluster_label = scipy.stats.mode(self.kmeans.fit_predict(features[y==0])).mode
        self.anomaly_cluster_label = 1 - normal_cluster_label


        # 4. Train RandomForest for fault classification
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        # We only train on fault data for classification
        fault_indices_train = y_train != 0
        self.rf_classifier.fit(X_train[fault_indices_train], y_train[fault_indices_train])

        fault_indices_test = y_test != 0
        y_pred = self.rf_classifier.predict(X_test[fault_indices_test])
        accuracy = accuracy_score(y_test[fault_indices_test], y_pred)
        print(f"随机森林模型准确率: {accuracy:.2f}")
        messagebox.showinfo("模型训练完成", f"随机森林模型准确率: {accuracy*100:.2f}%")


    def generate_fault_data(self, n_samples):
        fs = 1000  # Sampling frequency
        t = np.linspace(0, 1, fs, endpoint=False)
        X = []
        y = []

        for _ in range(n_samples):
            fault_type = np.random.randint(0, 4)
            y.append(fault_type)
            # Base signal
            signal = 1.0 * np.sin(2 * np.pi * 50 * t) # 50Hz fundamental
            noise = 0.5 * np.random.randn(len(t))
            
            if fault_type == 0: # Normal
                pass
            elif fault_type == 1: # Bearing wear
                # High-frequency vibrations
                signal += 0.8 * np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 250 * t)
            elif fault_type == 2: # Rotor imbalance
                # Increased fundamental amplitude
                signal = 2.5 * np.sin(2 * np.pi * 50 * t)
            elif fault_type == 3: # Loose anchor bolts
                # Sub-harmonics and sidebands
                signal += 0.7 * np.sin(2 * np.pi * 25 * t) + 0.4 * np.sin(2 * np.pi * 75 * t)
            
            X.append(signal + noise)
        return np.array(X), np.array(y)

    def extract_features(self, signal):
        fs = 1000
        # Time-domain features
        peak = np.max(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        crest_factor = peak / rms if rms != 0 else 0
        skewness = scipy.stats.skew(signal)
        kurtosis = scipy.stats.kurtosis(signal)
        
        # Frequency-domain features
        N = len(signal)
        yf = fft(signal)
        xf = np.linspace(0.0, 1.0/(2.0/fs), N//2)
        
        fft_vals = 2.0/N * np.abs(yf[0:N//2])
        
        main_freq_peak = np.max(fft_vals) if len(fft_vals) > 0 else 0
        
        # Frequencies of interest
        freq_50_idx = np.argmin(np.abs(xf - 50))
        freq_150_idx = np.argmin(np.abs(xf - 150))
        freq_25_idx = np.argmin(np.abs(xf - 25))

        f_50_amp = fft_vals[freq_50_idx]
        f_150_amp = fft_vals[freq_150_idx]
        f_25_amp = fft_vals[freq_25_idx]
        
        # Simple spectral energy
        spectral_energy = np.sum(fft_vals**2)
        
        return [peak, rms, crest_factor, skewness, kurtosis, main_freq_peak, f_50_amp, f_150_amp, f_25_amp, spectral_energy, np.mean(fft_vals), np.std(fft_vals)]


    def start_monitoring(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="状态: 正在运行", foreground="green")
        self.monitor_loop()

    def stop_monitoring(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 已停止", foreground="red")

    def monitor_loop(self):
        if not self.running:
            return

        # 1. Simulate real-time data
        new_signal, true_fault_type = self.generate_fault_data(1)
        new_signal = new_signal[0]
        true_fault_type = true_fault_type[0]

        # 2. Extract features and convert to NumPy array
        features = np.array(self.extract_features(new_signal)).reshape(1, -1)

        # 3. Anomaly Detection
        cluster = self.kmeans.predict(features)[0]
        
        status = "异常" if cluster == self.anomaly_cluster_label else "正常"
        fault_type_str = "N/A"
        probability_str = "N/A"

        # 4. Fault Classification if anomalous
        if status == "异常":
            fault_pred = self.rf_classifier.predict(features)[0]
            fault_proba = self.rf_classifier.predict_proba(features)
            fault_type_str = self.fault_types[fault_pred]
            probability = np.max(fault_proba)
            probability_str = f"{probability:.2%}"

        # 5. Update GUI
        self.update_plot(new_signal, status, fault_type_str, true_fault_type)
        self.update_results_tree(status, fault_type_str, probability_str)
        
        # 6. Schedule next update
        self.master.after(1000, self.monitor_loop)

    def update_plot(self, signal, status, fault_type_str, true_fault_type):
        self.ax[0].clear()
        self.ax[1].clear()
        
        t = np.linspace(0, 1, len(signal), endpoint=False)
        self.ax[0].plot(t, signal, color='blue' if status == "正常" else "red")
        self.ax[0].set_title(f"实时振动信号 - 真实: {self.fault_types[true_fault_type]} | 检测: {status} ({fault_type_str})")
        self.ax[0].set_xlabel("时间 (s)")
        self.ax[0].set_ylabel("振幅")
        self.ax[0].grid(True)

        fs = 1000
        N = len(signal)
        yf = fft(signal)
        xf = np.linspace(0.0, 1.0/(2.0/fs), N//2)
        fft_vals = 2.0/N * np.abs(yf[0:N//2])
        
        self.ax[1].plot(xf, fft_vals)
        self.ax[1].set_title("频域分析")
        self.ax[1].set_xlabel("频率 (Hz)")
        self.ax[1].set_ylabel("振幅")
        self.ax[1].set_xlim(0, fs/4) # Limit view to relevant frequencies
        self.ax[1].grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_results_tree(self, status, fault_type, probability):
        import datetime
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.results_tree.insert("", 0, values=(now, status, fault_type, probability))
        if len(self.results_tree.get_children()) > 20:
            self.results_tree.delete(self.results_tree.get_children()[-1])


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()