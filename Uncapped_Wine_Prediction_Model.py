import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
import os
import traceback

# --- Suppress common, harmless warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- CORE REGRESSION ANALYSIS FUNCTIONS ---

def format_poly(params, deg):
    """Formats polynomial coefficients into a readable string (y = c_n*x^n + ... + c_0)."""
    parts = []
    for i, c in enumerate(params):
        if abs(c) < 1e-15: continue
        term = f"{c:+.4e}"
        power = deg - i
        if power == 0:
            parts.append(term)
        elif power == 1:
            parts.append(f"{term}*x")
        else:
            parts.append(f"{term}*x^{power}")
    equation = 'y = ' + ' '.join(parts).lstrip('+')
    return equation.replace(' + -', ' - ').replace('+ ', '+').replace('- ', '-')


def sort_models(item):
    """Sort key: Exponential (0) before Polynomial (1), then by degree."""
    name = item[0]
    if name.startswith("Exponential"):
        return (0, name)
    if name.startswith("Polynomial (Deg "):
        try:
            return (1, int(name.split(' ')[2].strip(')'))
                    )
        except:
            return (2, name)
    return (3, name)


def get_best_fit(res, thresh=0.98):
    """Returns the simplest model (lowest degree) that meets the R2 threshold, or the highest R2 model found."""
    best_model = None
    best_r2 = -float('inf')

    for m_name, d in sorted(res.items(), key=sort_models):
        r2 = d['R-squared']
        if r2 > best_r2:
            best_r2 = r2
            best_model = (d['Prediction_Function'], m_name, r2, d['Equation'])
        if r2 >= thresh:
            return best_model
    return best_model if best_model else (None, "N/A", 0.0, "N/A")


def analyze_models(x, y, max_deg=6):
    """Performs Exponential and Polynomial regression analysis."""
    x, y = np.array(x, float), np.array(y, float)
    results = {}
    practical_max_deg = min(max_deg, len(x) - 1)
    STOP_THRESHOLD = 0.980
    analysis_stopped = False

    # 1. Exponential Analysis
    try:
        if np.all(y > 0):
            func = lambda t, a, b: a * np.exp(b * t)
            # Use safe initial guess (mean and small negative rate)
            p0_guess = [y.mean(), -0.01] if y.mean() > 0 else [1, -0.01]
            popt, _ = curve_fit(func, x, y, p0=p0_guess, maxfev=5000)
            exp_func = lambda t: popt[0] * np.exp(popt[1] * t)
            r2 = r2_score(y, exp_func(x))

            results['Exponential'] = {'R-squared': r2, 'Equation': f'y = {popt[0]:.4e} * exp({popt[1]:.4e} * x)',
                                      'Prediction_Function': exp_func}
            if r2 >= STOP_THRESHOLD: analysis_stopped = True
    except Exception:
        pass

    # 2. Polynomial Analysis
    if not analysis_stopped:
        for d in range(1, practical_max_deg + 1):
            if analysis_stopped: break
            try:
                coeffs = np.polyfit(x, y, d)
                p = np.poly1d(coeffs)
                r2 = r2_score(y, p(x))

                results[f"Polynomial (Deg {d})"] = {'R-squared': r2, 'Equation': format_poly(coeffs, d),
                                                    'Prediction_Function': p}
                if r2 >= STOP_THRESHOLD: analysis_stopped = True
            except:
                pass

    return results


def predict_scaled(func, init_sig, t):
    """Predicts the signal at time 't' using the Ratio Scaling method."""
    sig_0 = func(0.0)
    # Use ratio scaling if initial model value is non-zero, otherwise use absolute change
    val = init_sig * (func(t) / sig_0) if abs(sig_0) >= 1e-9 else init_sig + (func(t) - sig_0)
    return max(0.0, val)


# --- GUI APP CLASS ---

class WineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wine Oxygenation Analysis Model")
        self.geometry("650x750")

        self.opts = {k: tk.BooleanVar() for k in ["Equations", "Predictions", "Detailed", "All"]}
        self.main_opts = ["Equations", "Predictions", "Detailed"]
        self.results = {}
        self.filepath = None

        self._configure_styles()
        self.con = ttk.Frame(self)
        self.con.pack(fill="both", expand=True, padx=20, pady=20)
        self.show_welcome()

    def _configure_styles(self):
        """Configures ttk styles for a modern look."""
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TCheckbutton', font=('Arial', 12), relief='flat')
        s.configure('TCombobox', font=('Arial', 12))
        s.configure('TLabel', font=('Arial', 12))
        s.configure('TLabelframe', font=("Arial", 12, "bold"))
        s.configure('Header.TLabel', font=("Arial", 20, "bold"))

    def _clear_frame(self):
        """Destroys all children in the main frame."""
        for w in self.con.winfo_children(): w.destroy()

    # --- Option Logic ---
    def _toggle_all(self):
        """Synchronizes main options with the 'All' checkbox."""
        is_all_checked = self.opts["All"].get()
        for key in self.main_opts:
            self.opts[key].set(is_all_checked)

    def _check_individual_options(self):
        """Synchronizes 'All' based on the state of the three main options."""
        all_checked = all(self.opts[key].get() for key in self.main_opts)
        if all_checked != self.opts["All"].get():
            self.opts["All"].set(all_checked)

    def _validate_options(self):
        """Checks if at least one analysis option is selected."""
        if any(self.opts[k].get() for k in self.main_opts):
            self.show_upload()
        else:
            messagebox.showerror("Selection Error", "Please select at least one analysis option to continue.")

    # --- Screens ---
    def show_welcome(self):
        self._clear_frame()
        ttk.Label(self.con, text="Wine Oxygenation Analysis Model", style='Header.TLabel').pack(pady=(40, 20))
        # Introductory sentence and feature list are on separate lines
        desc = (" This model analyzes the effect of a wine being exposed to air over time.\n"
                "It is designed to analyze HS-SPME GC-MS data. \n"
                "The two features of the model are as follows: \n"
                "1) It determines the best fit regression model for a compound's corrected area over time.\n "
                "2) It can predict the corrected area for a previously measured compound, given time.\n")
        ttk.Label(self.con, text=desc, wraplength=600, justify="center").pack(pady=5)
        ttk.Label(self.con, text="This model was made using Gemini AI", font=("Arial", 10)).pack(pady=(20, 40))
        tk.Button(self.con, text="Continue", command=self.show_opts, font=("Arial", 12, "bold"), relief="flat",
                  bg="#4CAF50", width=25).pack(pady=10)

    def show_opts(self):
        self._clear_frame()
        ttk.Label(self.con, text="Choose Analysis Options", font=("Arial", 18, "bold")).pack(pady=(30, 20))

        frm = ttk.Frame(self.con)
        frm.pack(pady=10)

        options = [("Summary of Best-Fit Equations", "Equations"),
                   ("Enable Prediction Tool", "Predictions"),
                   ("Show Equation Analysis with R-Squared Values", "Detailed")]

        for text, var_key in options:
            ttk.Checkbutton(frm, text=text, variable=self.opts[var_key], command=self._check_individual_options).pack(
                fill="x", pady=5, anchor="w")

        ttk.Separator(frm).pack(fill="x", pady=10)
        ttk.Checkbutton(frm, text="All of the Above", variable=self.opts["All"], command=self._toggle_all).pack(
            fill="x", pady=5, anchor="w")

        tk.Button(self.con, text="Continue", command=self._validate_options, font=("Arial", 12, "bold"), relief="flat",
                  bg="#2196F3", width=25).pack(pady=20)

    def show_upload(self):
        self._clear_frame()
        ttk.Label(self.con, text="Data Selection", font=("Arial", 18, "bold")).pack(
            pady=(30, 10))
        ttk.Label(self.con,
                  text="The file must strictly follow the template format.",
                  foreground="red", wraplength=500).pack(pady=(0, 15))

        tk.Button(self.con, text="Download Template (.xlsx Format)", command=self.dl_temp, font=("Arial", 12, "bold"),
                  relief="flat", bg="#DCDCDC", width=40).pack(pady=10)
        ttk.Label(self.con, text="- OR -", font=("Arial", 10)).pack(pady=5)

        tk.Button(self.con, text="Upload Data Sheet (.xlsx or .csv)", command=self.ul_file, font=("Arial", 12, "bold"),
                  relief="flat", bg="#DCDCDC", width=25).pack(pady=10)

        status_text = f"No file selected"
        status_color = "red"
        if self.filepath:
            status_text = f"Loaded: {os.path.basename(self.filepath)}"
            status_color = "green"

        self.status_lbl = ttk.Label(self.con, text=status_text, font=("Arial", 10), foreground=status_color)
        self.status_lbl.pack(pady=(0, 10))

        run_state = "normal" if self.filepath else "disabled"
        run_bg = "#2196F3" if self.filepath else "#FF9800"

        self.run_btn = tk.Button(self.con, text="Run Analysis", command=self.run_analysis,
                                 font=("Arial", 12, "bold"), relief="flat", bg=run_bg, state=run_state, width=25)
        self.run_btn.pack(pady=20)

    # --- File/Data Handlers ---
    def dl_temp(self):
        """Generates a true Excel file in the requested format and attempts a local save."""
        f = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save Kinetic Analysis Template"
        )

        if not f: return

        num_compounds = 5
        time_points = [0, 45, 90, 135, 180, 270, 360]
        header = []
        for i in range(1, num_compounds + 1):
            header.extend([f"Compound {i} Name", "Minutes", "Corr. Area"])

        rows = []
        for time in time_points:
            row = []
            for c_idx in range(num_compounds):
                # Only put the compound name placeholder in the first row
                compound_name = f"Compound {c_idx + 1} Example" if time == 0 else ""
                area_placeholder = None
                row.extend([compound_name, time, area_placeholder])
            rows.append(row)

        try:
            df_template = pd.DataFrame(rows, columns=header)
            # xlsxwriter is now imported directly
            df_template.to_excel(f, index=False, header=True, engine='xlsxwriter')

            messagebox.showinfo("Save Attempted",
                                f"Excel template save attempted to: {f}\n\n"
                                "If the file did not appear on your local system, it is because this execution "
                                "environment prevents direct saving. You may need to ask for the browser download method.")

        except Exception as e:
            msg = f"Error attempting local file save. The environment likely prevents file writing.\n\nDetail: {e}"
            messagebox.showerror("File Save Error", msg)

    def ul_file(self):
        f = filedialog.askopenfilename(filetypes=[("Data Files", "*.xlsx *.csv")])
        if f and os.path.splitext(f)[1].lower() in ['.xlsx', '.csv']:
            self.filepath = f
            self.status_lbl.config(text=f"Loaded: {os.path.basename(f)}", foreground="green")
            self.run_btn.config(state="normal", bg="#2196F3")
        else:
            self.filepath = None
            self.status_lbl.config(text="No file selected (Invalid/No file)", foreground="red")
            self.run_btn.config(state="disabled", bg="#FF9800")

    def _revert_to_upload(self, error_message):
        """Displays error, clears file state, and reverts to the upload screen (STRICT REJECTION)."""
        self.filepath = None
        self.results = {}
        messagebox.showerror("File Rejected", error_message)
        self.show_upload()

    def run_analysis(self):
        if not self.filepath: return

        try:
            is_excel = self.filepath.lower().endswith('.xlsx')
            reader = pd.read_excel if is_excel else pd.read_csv

            read_kwargs = {'header': 0}
            if not is_excel: read_kwargs['low_memory'] = False

            df_raw = reader(self.filepath, **read_kwargs)
            df_raw.dropna(how='all', inplace=True)

            if df_raw.empty or df_raw.shape[0] < 1:
                return self._revert_to_upload("Structural Error: File appears empty or corrupted.")

            df_cols = df_raw.columns.tolist()
            compound_groups = []
            i = 0
            # Identify groups of three columns (Compound Name, Minutes, Corr. Area)
            while i < len(df_cols):
                if 'Compound' in df_cols[i] and i + 2 < len(df_cols):
                    # Check for expected headers in columns i+1 and i+2
                    if any(h in df_cols[i + 1] for h in ['Min', 'Time', 'T']) and any(
                            h in df_cols[i + 2] for h in ['Area', 'Sig']):
                        compound_groups.append((df_cols[i], df_cols[i + 1], df_cols[i + 2]))
                    i += 3
                else:
                    i += 1

            if not compound_groups:
                return self._revert_to_upload(
                    "Structural Error: Could not find any compound triplets (e.g., 'Compound X Name', 'Minutes', 'Corr. Area').")

            all_compound_data = {}
            for name_col, time_col, area_col in compound_groups:
                # Efficiently retrieve the compound name (first non-NaN value in the name column, or default to column header)
                compound_series = df_raw[name_col].dropna()
                compound_name = name_col # Default to column header
                if not compound_series.empty:
                    compound_name = str(compound_series.iloc[0]).strip()

                # Extract and clean data points
                temp_df = df_raw[[time_col, area_col]].dropna(how='any').copy()

                if temp_df.empty:
                    print(f"Skipping compound {compound_name}: No valid data points.")
                    continue

                try:
                    # Coerce to float using vectorized numpy operations for efficiency
                    x = temp_df[time_col].astype(float).values
                    y = temp_df[area_col].astype(float).values
                except ValueError:
                    return self._revert_to_upload(
                        f"Non-Numeric Data Error for Compound '{compound_name}'. Check values in '{time_col}' or '{area_col}' columns for non-numeric entries.")

                if len(x) < 2:
                    return self._revert_to_upload(
                        f"Data Insufficiency Error for Compound '{compound_name}'. Requires minimum two valid data points. Found {len(x)}.")

                all_compound_data[compound_name] = (x, y)

            # Run Regression.
            if not all_compound_data:
                return self._revert_to_upload(
                    "Structural Error: No compound data sets could be processed successfully.")

            # Perform regression analysis for all compounds
            self.results = {name: analyze_models(x, y) for name, (x, y) in all_compound_data.items()}
            self.show_results()

        except pd.errors.ParserError as e:
            self._revert_to_upload(f"File Read Error. Check file validity: {e}")
        except Exception as e:
            self._revert_to_upload(f"An unexpected critical error occurred: {e}")
            traceback.print_exc()

    # --- Results and Prediction Screens ---
    def calculate_dynamic_widths(self, results):
        """Calculates optimal column widths for formatting the results table."""
        # Max compound and model name lengths, plus padding
        max_compound_len = max(len("COMPOUND"), *[len(n) for n in results])
        max_model_len = max(len("BEST MODEL"), *[len(get_best_fit(r)[1]) for r in results.values()])

        padding = 6
        return max_compound_len + padding, max_model_len + padding, len("R^2") + 6, len("EQUATION") + 6

    def show_results(self):
        self._clear_frame()
        ttk.Label(self.con, text="Analysis Results Summary", font=("Arial", 18, "bold")).pack(pady=(10, 5))

        # --- MODIFICATION 1: Conditionalize the creation of the scrolledtext widget ---
        should_show_analysis = self.opts["Equations"].get() or self.opts["Detailed"].get()

        if should_show_analysis:
            compound_w, model_w, r2_w, equation_w = self.calculate_dynamic_widths(self.results)
            total_width = compound_w + model_w + r2_w + equation_w + 5

            txt = scrolledtext.ScrolledText(self.con, width=total_width, height=28, font=("Courier New", 10), wrap="none")
            txt.pack(pady=5, fill="both", expand=True, padx=10)

            out = ""
            format_str = f"{{:<{compound_w}}}{{:<{model_w}}}{{:<{r2_w}}}{{:<}}\n"

            # 1. Equations Summary (Best Fit Only)
            if self.opts["Equations"].get():
                out += "\n--- Best Fit Equations Summary ---\n\n"
                out += format_str.format("COMPOUND", "BEST MODEL", "R^2", "EQUATION")
                out += "-" * total_width + "\n"

                for n, r in self.results.items():
                    _, m, r2, eq = get_best_fit(r)
                    out += format_str.format(n, m, f"{r2:.5f}", eq)
                out += "\n"

            # 2. Detailed Model Analysis (All Models)
            if self.opts["Detailed"].get():
                out += "\n--- Detailed Model Analysis ---\n"
                for n, r in self.results.items():
                    out += f"\n{n}:\n"
                    for m, d in sorted(r.items(), key=sort_models):
                        out += f"  {m:<{model_w - 4}}| R^2={d.get('R-squared', 0):.5f} | {d.get('Equation', '')}\n"

            txt.insert(tk.END, out)
            txt.config(state="disabled")

        if self.opts["Predictions"].get():
            self._build_pred_ui()

        # Footer Buttons
        bf = ttk.Frame(self.con)
        bf.pack(pady=20, fill="x")
        tk.Button(bf, text="New Analysis", command=self.show_opts, font=("Arial", 11), bg="#DCDCDC").pack(side="left",
                                                                                                          padx=20)
        tk.Button(bf, text="Done (Go to Final Screen)", command=self.show_done, font=("Arial", 11), bg="#4CAF50").pack(
            side="right", padx=20)

    def _build_pred_ui(self):
        fr = ttk.LabelFrame(self.con, text="Prediction Tool", padding=(10, 5))
        fr.pack(fill="x", pady=10, padx=10)
        fr.grid_columnconfigure((0, 2), weight=1)
        fr.grid_columnconfigure((1, 3), weight=2)

        ttk.Label(fr, text="Compound:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cmb = ttk.Combobox(fr, values=list(self.results.keys()), width=18, state='readonly', font=("Arial", 12))
        self.cmb.grid(row=0, column=1, padx=5, sticky="ew")
        if self.results: self.cmb.current(0)

        ttk.Label(fr, text="Initial Signal:").grid(row=0, column=2, padx=5, sticky="w")
        self.e_sig = ttk.Entry(fr, width=10, font=("Arial", 12))
        self.e_sig.grid(row=0, column=3, padx=5, sticky="ew")

        ttk.Label(fr, text="Aging Time (min):").grid(row=1, column=0, padx=5, sticky="w")
        self.e_tim = ttk.Entry(fr, width=10, font=("Arial", 12))
        self.e_tim.grid(row=1, column=1, padx=5, sticky="ew")

        tk.Button(fr, text="Calculate Prediction", bg="#4CAF50", fg="black", command=self.calc_pred,
                  font=("Arial", 12, "bold"), relief="flat").grid(row=1, column=2, columnspan=2, sticky="ew", padx=5,
                                                                  pady=5)

        self.res_lbl = ttk.Label(fr, text="Ready for prediction...", foreground="blue", justify="left", wraplength=550)
        self.res_lbl.grid(row=2, column=0, columnspan=4, pady=5, padx=5, sticky="ew")

    def calc_pred(self):
        try:
            comp = self.cmb.get()
            initial_signal = float(self.e_sig.get().strip())
            time = float(self.e_tim.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Invalid Input. Signal and Time must be numbers.")
            return

        try:
            func, mod, r2, eq = get_best_fit(self.results[comp])

            if func:
                val = predict_scaled(func, initial_signal, time)
                res_text = (f"Predicted Signal after {time} min: {val:,.2f} corrected area\n"
                            f"Model Used: {mod} (R^2={r2:.5f})\n"
                            f"Equation: {eq}")
                self.res_lbl.config(text=res_text, foreground="black")
            else:
                self.res_lbl.config(text="Error: No valid model found for selected compound.", foreground="red")
        except KeyError:
            messagebox.showerror("Error", "Compound not found in results.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")

    def show_done(self):
        self._clear_frame()
        # --- MODIFICATION 2: Horizontal centering of text ---
        ttk.Label(self.con, text="Analysis Complete!", font=("Arial", 24, "bold"), justify="center").pack(pady=(80, 40))
        ttk.Label(self.con, text="Thank you for using the model!", font=("Arial", 16), justify="center").pack(
            pady=20)
        tk.Button(self.con, text="Close Window", command=self.destroy, bg="#f44336", fg="black", width=30,
                  font=("Arial", 12, "bold"), relief="flat").pack(pady=20)


if __name__ == "__main__":
    WineApp().mainloop()