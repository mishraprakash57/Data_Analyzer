import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker as mticker

# ----------------------------------------
# Module 1: Monthly Average Plot with Consistent Axis Scaling
# ----------------------------------------
def process_files(file_paths):
    processed_data = []
    max_val = 0  # Variable to hold the maximum value for scaling

    # First pass: find the maximum value of both RPRO and PAL across all files
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        city_name = file_name[:-9]  # Extract city name
        year = file_name[-9:-5]     # Extract year
        
        data = pd.read_excel(file_path)
        cleaned_data = data[(data['PAL'] > 0) & (data['RPRO'] > 0)]
        Q1 = cleaned_data[['PAL', 'RPRO']].quantile(0.25)
        Q3 = cleaned_data[['PAL', 'RPRO']].quantile(0.75)
        IQR = Q3 - Q1
        cleaned_data = cleaned_data[~((cleaned_data[['PAL', 'RPRO']] < (Q1 - 3 * IQR)) | 
                                      (cleaned_data[['PAL', 'RPRO']] > (Q3 + 3 * IQR))).any(axis=1)]
        
        # Find the maximum value between PAL and RPRO and update max_val
        max_val = max(max_val, cleaned_data['PAL'].max(), cleaned_data['RPRO'].max())
        
        coefficients = np.polyfit(cleaned_data['RPRO'], cleaned_data['PAL'], 1)
        trend_line = np.poly1d(coefficients)

        predictions = trend_line(cleaned_data['RPRO'])
        rmse = np.sqrt(mean_squared_error(cleaned_data['PAL'], predictions))
        r_squared = r2_score(cleaned_data['PAL'], predictions)
        
        processed_data.append({
            'city_name': city_name,
            'year': year,
            'data': cleaned_data,
            'trend_line': trend_line,
            'equation': f'y = {coefficients[0]:.2f} * x + {coefficients[1]:.2f}',
            'rmse': rmse,
            'r_squared': r_squared
        })

    return processed_data, max_val  # Return processed data and max value for scaling

def plot_data(data_info, max_val):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Scatter plot (PAL on y-axis, RPRO on x-axis)
    ax.scatter(data_info['data']['RPRO'], data_info['data']['PAL'], alpha=0.7, edgecolors='k')
    
    # Trend line (RPRO on x-axis, PAL on y-axis)
    ax.plot(data_info['data']['RPRO'], data_info['trend_line'](data_info['data']['RPRO']), color='red')
    
    # Diagonal line (y = x) fully touching the edge of the plot
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--')
    
    # Ensure equal axis limits for both axes to match grid distances
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    
    # Set major grid on both axes and enforce equal grid spacing
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Make sure the grid is visible and consistent
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    
    # Title and labels
    ax.set_title(f"{data_info['city_name']}, {data_info['year']}")
    ax.set_xlabel('RPRO')
    ax.set_ylabel('PAL')

    # Display the trend line equation, RMSE, and R-squared on the plot
    ax.text(0.05, 0.9, f'Equation: {data_info["equation"]}', transform=ax.transAxes, fontsize=10, color='red')
    ax.text(0.05, 0.85, f'RMSE: {data_info["rmse"]:.2f}', transform=ax.transAxes, fontsize=10, color='blue')
    ax.text(0.05, 0.80, f'R²: {data_info["r_squared"]:.2f}', transform=ax.transAxes, fontsize=10, color='green')
    
    return fig

def show_all_plots(processed_data, max_val):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    num_files = len(processed_data)
    num_columns = 2  # Display two plots per row
    num_rows = (num_files + 1) // num_columns
    
    for i, data_info in enumerate(processed_data):
        fig = plot_data(data_info, max_val)
        canvas_plot = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_plot.draw()
        row = i // num_columns
        col = i % num_columns
        canvas_plot.get_tk_widget().grid(row=row*2, column=col, padx=10, pady=10)
        
        save_button = tk.Button(plot_frame, text=f"Save {data_info['city_name']}, {data_info['year']} Plot",
                                command=lambda fig=fig, city=data_info['city_name'], year=data_info['year']: save_plot(fig, city, year))
        save_button.grid(row=row*2 + 1, column=col, pady=5)

def select_files_module1():
    file_paths = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xlsx")])
    if file_paths:
        processed_data, max_val = process_files(file_paths)  # Get processed data and global max value
        show_all_plots(processed_data, max_val)  # Pass max_val to ensure same scale for all plots

def save_plot(fig, city_name, year):
    save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                             filetypes=[("PNG Files", "*.png")],
                                             initialfile=f"{city_name}_{year}_plot.png")
    if save_path:
        fig.savefig(save_path)
        messagebox.showinfo("Save Plot", f"Plot saved successfully at {save_path}")

# ----------------------------------------
# Module 2: APD% Plot functions
# ----------------------------------------
def save_plot_apd(fig, plot_name):
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")],
                                             initialfile=f"{plot_name}.png")
    if save_path:
        fig.savefig(save_path)
        messagebox.showinfo("Success", f"{plot_name} has been saved as {save_path}")

def load_and_process_file_apd():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return
    
    try:
        excel_data = pd.ExcelFile(file_path)
        sheet_names = excel_data.sheet_names  # These are the months
        
        apd_data = pd.DataFrame()

        for sheet in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            if 'APD%' in df.columns:
                df['Month'] = sheet
                apd_data = pd.concat([apd_data, df[['City', 'APD%', 'Month']]], ignore_index=True)
        
        visualize_results_apd(apd_data)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def visualize_results_apd(apd_data):
    for widget in plot_frame.winfo_children():
        widget.destroy()

    bar_plot_frame = tk.Frame(plot_frame)
    bar_plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=20)

    # Increase the figure size for better visibility of labels
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='City', y='APD%', hue='Month', data=apd_data, ax=ax1)
    ax1.set_title('APD% by City Across Months')

    # Rotate the city names on the Y-axis for better visibility
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Adjust spacing to prevent label clipping
    plt.tight_layout()

    canvas1 = FigureCanvasTkAgg(fig1, master=bar_plot_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    download_button1 = tk.Button(bar_plot_frame, text="Download Bar Plot", command=lambda: save_plot_apd(fig1, "Bar Plot"))
    download_button1.pack(side=tk.TOP, pady=10)

    heatmap_frame = tk.Frame(plot_frame)
    heatmap_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=20)

    heatmap_data = apd_data.pivot(index='City', columns='Month', values='APD%')

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=.5, ax=ax2)
    ax2.set_title('APD% Heatmap Across Cities and Months')

    canvas2 = FigureCanvasTkAgg(fig2, master=heatmap_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    download_button2 = tk.Button(heatmap_frame, text="Download Heatmap", command=lambda: save_plot_apd(fig2, "Heatmap"))
    download_button2.pack(side=tk.TOP, pady=10)

# ----------------------------------------
# Module 3: Monthly Average Analysis
# ----------------------------------------
def process_file_module3(file_path):
    df = pd.read_excel(file_path)
    df.rename(columns={'Day': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.strftime('%B')
    monthly_avg = df.groupby('Month')[['PAL', 'RPRO']].mean().reset_index()
    monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=[
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'], ordered=True)
    monthly_avg = monthly_avg.sort_values('Month')
    return monthly_avg

def plot_data_module3(file_paths, frame):
    for widget in frame.winfo_children():
        widget.destroy()  # Clear the previous plots

    for i, file_path in enumerate(file_paths):
        monthly_avg = process_file_module3(file_path)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(monthly_avg['Month'], monthly_avg['PAL'], label='PAL', marker='o')
        ax.plot(monthly_avg['Month'], monthly_avg['RPRO'], label='RPRO', marker='o')
        string_val= f"{file_path.split('/')[-1]}"
        string_val=string_val.replace(".xlsx","")
        #ax.set_title(f"{file_path.split('/')[-1]}")
        ax.set_title(string_val)
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Value')
        ax.set_ylim([0, 8])
        ax.legend()
        ax.set_xticklabels(monthly_avg['Month'], rotation=30)

        # Embed the plot into the tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=(i//2)*2, column=i%2, padx=10, pady=10)  # 2 plots per row

        # Download button for each plot
        def save_plot(fig=fig, file_path=file_path):
            save_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG files", "*.png")])
            if save_path:
                fig.savefig(save_path)

        download_button = tk.Button(frame, text="Download Plot", command=lambda fig=fig: save_plot(fig))
        download_button.grid(row=(i//2)*2+1, column=i%2, padx=10, pady=10)  # Button directly below the plot

def upload_files_module3():
    file_paths = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xlsx")])
    plot_data_module3(file_paths, plot_frame)

# ----------------------------------------
# Main GUI that integrates all modules
# ----------------------------------------
def launch_gui():
    global plot_frame
    
    root = tk.Tk()
    root.title("Data Analysis Modules")                                                                                          
    label = tk.Label(root, text="Select an Option:", font=('Helvetica', 16))
    label.pack(pady=10)

    def launch_module1():
        global processed_data
        processed_data = []
        select_files_module1()

    def launch_module2():
        load_and_process_file_apd()

    def launch_module3():
        upload_files_module3()

    # Add buttons for each module
    btn_module1 = tk.Button(root, text="Monthly Trend Plot (Module 1)", command=launch_module1, height=2, width=30)
    btn_module1.pack(pady=10)

    btn_module2 = tk.Button(root, text="Absolute Percentage Difference (Module 2)", command=launch_module2, height=2, width=30)
    btn_module2.pack(pady=10)

    btn_module3 = tk.Button(root, text="Monthly Average Analysis (Module 3)", command=launch_module3, height=2, width=30)
    btn_module3.pack(pady=10)

    # Add Reset button
    btn_reset = tk.Button(root, text="Reset Plots", command=reset_plots, height=2, width=30)
    btn_reset.pack(pady=10)

    # Create a scrollable canvas for displaying the plots
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create a frame inside the scrollable canvas to hold the plots
    global plot_frame
    plot_frame = tk.Frame(scrollable_frame)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    root.mainloop()

def reset_plots():
    for widget in plot_frame.winfo_children():
        widget.destroy()

# Run the GUI application
if __name__ == "__main__":
    launch_gui()



