from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os

# Sample dictionary data
data = {
    "Title": "Sample PDF Report",
    "Author": "John Doe",
    "Summary": "This is a sample summary of the report.",
    "Data": {
        "A": 10,
        "B": 15,
        "C": 7,
        "D": 5,
        "E": 12
    }
}

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, data["Title"], 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln()

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_plot(self, img_path):
        self.image(img_path, x=10, y=None, w=180)

    def add_table(self, table_data):
        self.set_font("Arial", "B", 12)
        self.cell(50, 10, "Key", 1, 0, "C")
        self.cell(50, 10, "Value", 1, 1, "C")
        self.set_font("Arial", "", 12)
        for key, value in table_data.items():
            self.cell(50, 10, str(key), 1, 0, "C")
            self.cell(50, 10, str(value), 1, 1, "C")
        self.ln()

# Create a plot and save to a temporary file
fig, ax = plt.subplots()
ax.bar(data["Data"].keys(), data["Data"].values())
ax.set_xlabel("Categories")
ax.set_ylabel("Values")
ax.set_title("Sample Data Plot")

# Use tempfile to save the plot temporarily
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
    plt.savefig(temp_file.name, format="PNG")
    temp_file_path = temp_file.name

# Generate PDF
pdf = PDF()

pdf.add_page()
pdf.chapter_title("Author")
pdf.chapter_body(data["Author"])
pdf.chapter_title("Summary")
pdf.chapter_body(data["Summary"])

pdf.chapter_title("Data Table")
pdf.add_table(data["Data"])

pdf.chapter_title("Plot")
pdf.add_plot(temp_file_path)

output_path = "sample_report.pdf"
pdf.output(output_path)

# Clean up the temporary file
os.remove(temp_file_path)

print(f"PDF saved to {output_path}")