import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src_updt import main

class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Configuring evolutionary algorithm")
        self.root.geometry("550x600")
        self.root.resizable(True, True)

        self.row_widgets = []

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.root, borderwidth=0, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.main_frame = ttk.Frame(self.canvas)
        self.canvas_window_id = self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        self.main_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.optimization_type = tk.StringVar(value="min")
        self.function_name = tk.StringVar(value="Sphere")
        self.num_vars = tk.IntVar(value=2)
        self.num_epochs = tk.IntVar(value=200)
        self.num_pop = tk.IntVar(value=100)

        self.selection_method = tk.StringVar(value="best")
        self.selection_proportion = tk.DoubleVar(value=0.6)
        self.tournament_k = tk.IntVar(value=3)
        self.tournament_num = tk.IntVar(value=20)

        self.crossover_method = tk.StringVar(value="onepoint")

        self.mutation_method = tk.StringVar(value="onepoint")
        self.mutation_rate = tk.DoubleVar(value=0.1)

        self.inversion_rate = tk.DoubleVar(value=0.1)

        self.elitism = tk.BooleanVar(value=True)

        self.bounds_entries = []
        self.same_for_all = tk.BooleanVar(value=False)
        self.common_low = tk.StringVar(value="-5.12")
        self.common_high = tk.StringVar(value="5.12")
        self.common_precision = tk.StringVar(value="6")

        self.create_widgets()

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window_id, width=event.width)

    def create_widgets(self):
        frame1_top = ttk.LabelFrame(self.main_frame)
        frame1_top.pack(anchor="w", padx=10)

        ttk.Label(frame1_top, text="Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(frame1_top, text="min", variable=self.optimization_type, value="min").grid(row=0, column=1, padx=0, pady=5)
        ttk.Radiobutton(frame1_top, text="max", variable=self.optimization_type, value="max").grid(row=0, column=2, padx=0, pady=5)

        frame2_top = ttk.LabelFrame(self.main_frame)
        frame2_top.pack(anchor="w", padx=10)

        ttk.Label(frame2_top, text="Function:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        func_combo = ttk.Combobox(frame2_top, textvariable=self.function_name, values=["Sphere", "Hyperellipsoid", "Schwefel", "Ackley", "Michalewicz", "Rastrigin", "Rosenbrock", "Dejong3"], state="readonly", width=15)
        func_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        frame_vars = ttk.LabelFrame(self.main_frame)
        frame_vars.pack(anchor="w", padx=10)

        ttk.Label(frame_vars, text="Num. variables:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        spin_vars = ttk.Spinbox(frame_vars, from_=1, to=20, textvariable=self.num_vars, command=self.update_bounds_table, width=5)
        spin_vars.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        spin_vars.bind("<KeyRelease>", lambda e: self.update_bounds_table())

        self.table_frame = ttk.Frame(self.main_frame)
        self.table_frame.pack(fill="both", padx=10, pady=5, expand=True)
        self.update_bounds_table()

        frame_params = ttk.LabelFrame(self.main_frame)
        frame_params.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_params, text="Num. popul:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_params, textvariable=self.num_pop, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_params, text="Num. epochs:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_params, textvariable=self.num_epochs, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_params, text="Selection technique:").grid(row=2, column=0, padx=5, sticky="w")
        sel_combo = ttk.Combobox(frame_params, textvariable=self.selection_method, values=["best", "worst", "ranking", "roulette", "random", "tournament"], state="readonly", width=15)
        sel_combo.grid(row=2, column=1, padx=5, sticky="w")
        sel_combo.bind("<<ComboboxSelected>>", self.toggle_selection_params)

        self.sel_params_frame = ttk.Frame(frame_params)
        self.sel_params_frame.grid(row=2, column=3, padx=5, sticky="w")
        self.toggle_selection_params()

        ttk.Label(frame_params, text="Crossover technique:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        cross_combo = ttk.Combobox(frame_params, textvariable=self.crossover_method, values=["shuffle", "onepoint", "twopoint", "threepoint", "uniform", "replacement", "grain", "devastation"], state="readonly", width=15)
        cross_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_params, text="Mutation technique:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        mut_combo = ttk.Combobox(frame_params, textvariable=self.mutation_method, values=["onepoint", "twopoint", "edge"], state="readonly", width=15)
        mut_combo.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_params, text="Mutation rate:").place(x=252, y=147)
        ttk.Entry(frame_params, textvariable=self.mutation_rate, width=8).place(x=337, y=147)

        ttk.Label(frame_params, text="Inversion rate:").place(x=252, y=179)
        ttk.Entry(frame_params, textvariable=self.inversion_rate, width=8).place(x=337, y=179)

        ttk.Checkbutton(frame_params, text="Elitism", variable=self.elitism).grid(row=5, column=0, padx=5, pady=5, sticky="w")

        self.bottom_frame = ttk.LabelFrame(self.main_frame)
        self.bottom_frame.pack(anchor="w", padx=10)

        btn_run = ttk.Button(self.main_frame, text="Launch algorithm", command=self.run_simulation)
        btn_run.pack(pady=10)

    def update_bounds_table(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        n = self.num_vars.get()
        if n < 1:
            return

        headers = ["Num. var.", "Lower bound", "Upper bound", "Number sig. fig."]
        for col, text in enumerate(headers):
            lbl = ttk.Label(self.table_frame, text=text, font=("Arial", 10, "bold"))
            lbl.grid(row=0, column=col, padx=5, pady=2, sticky="w")
            self.table_frame.grid_columnconfigure(col, weight=1, uniform="table_col")

        self.row_widgets = []
        for i in range(1, n+1):
            lbl = ttk.Label(self.table_frame, text=str(i))
            lbl.grid(row=i, column=0, padx=5, pady=2, sticky="w")

            low = ttk.Entry(self.table_frame, width=12)
            low.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            low.insert(0, "-5.12")

            high = ttk.Entry(self.table_frame, width=12)
            high.grid(row=i, column=2, padx=5, pady=2, sticky="ew")
            high.insert(0, "5.12")

            prec = ttk.Entry(self.table_frame, width=8)
            prec.grid(row=i, column=3, padx=5, pady=2, sticky="ew")
            prec.insert(0, "6")

            self.row_widgets.append((lbl, low, high, prec))

        common_frame = ttk.Frame(self.table_frame)
        common_frame.grid(row=n + 1, column=0, columnspan=4, sticky="ew", pady=5)

        ttk.Checkbutton(common_frame, text="Same for all variables", variable=self.same_for_all, command=self.toggle_same_for_all).pack(side="left", padx=5)

        self.toggle_same_for_all()

    def toggle_same_for_all(self):
        checked = self.same_for_all.get()
        if checked:
            for idx, (lbl, low, high, prec) in enumerate(self.row_widgets):
                if idx == 0:
                    low.config(state='normal')
                    high.config(state='normal')
                    prec.config(state='normal')
                else:
                    lbl.grid_remove()
                    low.grid_remove()
                    high.grid_remove()
                    prec.grid_remove()
        else:
            for idx, (lbl, low, high, prec) in enumerate(self.row_widgets):
                lbl.grid()
                low.grid()
                high.grid()
                prec.grid()
                if idx == 0:
                    continue
                else:
                    first_low, first_high, first_prec = self.row_widgets[0][1:]
                    low.delete(0, tk.END)
                    low.insert(0, first_low.get())
                    high.delete(0, tk.END)
                    high.insert(0, first_high.get())
                    prec.delete(0, tk.END)
                    prec.insert(0, first_prec.get())
                    low.config(state='normal')
                    high.config(state='normal')
                    prec.config(state='normal')

    def toggle_selection_params(self, event=None):
        for widget in self.sel_params_frame.winfo_children():
            widget.destroy()

        self.sel_params_frame.config(width=400, height=80)
        self.sel_params_frame.grid_propagate(False)

        method = self.selection_method.get()
        if method == "tournament":
            ttk.Label(self.sel_params_frame, text="K:").place(x=0, y=30)
            ttk.Entry(self.sel_params_frame, textvariable=self.tournament_k, width=5).place(x=55, y=30)
            ttk.Label(self.sel_params_frame, text="N. tour.:").place(x=0, y=60)
            ttk.Entry(self.sel_params_frame, textvariable=self.tournament_num, width=5).place(x=55, y=60)
        else:
            ttk.Label(self.sel_params_frame, text="Proport.:").place(x=0, y=30)
            ttk.Entry(self.sel_params_frame, textvariable=self.selection_proportion, width=5).place(x=55, y=30)

    def run_simulation(self):
        try:
            n_vars = self.num_vars.get()
            if n_vars <= 0:
                raise ValueError("Number of variables should be positive")

            low, high, num_significant_figures = [], [], []
            if not self.same_for_all.get():
                for row in self.row_widgets:
                    low_val = float(row[1].get())
                    high_val = float(row[2].get())
                    prec_val = int(row[3].get())
                    if low_val >= high_val:
                        raise ValueError("Lower bound should be less than the upper")
                    if prec_val < 1:
                        raise ValueError("Number of significant figures should be >= 1")
                    low.append(low_val)
                    high.append(high_val)
                    num_significant_figures.append(prec_val)
            else:
                row0 = self.row_widgets[0]
                low.extend(int(self.num_vars.get())*[float(row0[1].get())])
                high.extend(int(self.num_vars.get())*[float(row0[2].get())])
                num_significant_figures.extend(int(self.num_vars.get())*[int(row0[3].get())])

            num_population = self.num_pop.get()
            num_epochs = self.num_epochs.get()
            if num_epochs <= 0:
                raise ValueError("Number of epochs should be positive")

            sel_method = self.selection_method.get()
            k = 0
            number_tournaments = 0
            percentage_keep = 0
            if sel_method == "tournament":
                k = self.tournament_k.get()
                number_tournaments = self.tournament_num.get()
                if k <= 0 or number_tournaments <= 0:
                    raise ValueError("Parameters of selection method should be positive")
            else:
                percentage_keep = self.selection_proportion.get()
                if not (0 < percentage_keep <= 1):
                    raise ValueError("Proportion should be between 0 and 1")

            cross_type = self.crossover_method.get()
            mutation_type = self.mutation_method.get()
            mutation_rate = self.mutation_rate.get()
            if not (0 <= mutation_rate <= 1):
                raise ValueError("Mutation rate should be between 0 and 1")
            inversion_rate = self.inversion_rate.get()
            if not (0 <= inversion_rate <= 1):
                raise ValueError("Inversion rate should be between 0 and 1")

            elitarism_bool = self.elitism.get()
            opt_type = self.optimization_type.get()
            function_name = self.function_name.get()

            if sel_method == "tournament":
                point, fun_value, target_values, iteration_times = main(function_name, opt_type, n_vars, num_population, num_epochs, sel_method, low, high, num_significant_figures, mutation_rate, inversion_rate, cross_type, mutation_type, elitarism_bool, k=k, number_tournaments=number_tournaments) 

            if sel_method != "tournament":
                point, fun_value, target_values, iteration_times = main(function_name, opt_type, n_vars, num_population, num_epochs, sel_method, low, high, num_significant_figures, mutation_rate, inversion_rate, cross_type, mutation_type, elitarism_bool, percentage_keep=percentage_keep) 

            self.show_plots(range(1, len(iteration_times)+1), target_values, iteration_times, function_name)

            for widget in self.bottom_frame.winfo_children():
                widget.destroy()

            ttk.Label(self.bottom_frame, text=f"Optimal y: {fun_value}").grid(row=0, column=0, sticky="w")
            optimal_point_string = ", ".join(str(i) for i in point)
            ttk.Label(self.bottom_frame, text=f"Optimal y at point x: (" + optimal_point_string + ")").grid(row=1, column=0, sticky="w")
        except ValueError as e:
            messagebox.showerror("Error in input", str(e))
        

    def show_plots(self, iterations, fitness, times, func_name):
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Results of simulation: {func_name}")
        plot_window.geometry("800x600")
        plot_window.grid_rowconfigure(0, weight=1)
        plot_window.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(plot_window, borderwidth=0, highlightthickness=0)
        v_scroll = tk.Scrollbar(plot_window, orient="vertical", command=canvas.yview)
        h_scroll = tk.Scrollbar(plot_window, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(iterations, fitness, 'b-')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Function value")
        ax1.set_title(f"Function {func_name} value at each iteration")
        ax1.grid(True)

        ax2.plot(iterations, times, 'r-')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Time (s)")
        ax2.set_title("Time of an iteration")
        ax2.grid(True)

        plt.tight_layout()

        fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(fill="both", expand=True)

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner_frame.bind("<Configure>", on_frame_configure)

        def on_canvas_configure(event):
            canvas.itemconfig(1, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()