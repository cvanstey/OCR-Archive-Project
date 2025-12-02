# For Google Colab - Install dependencies first:
# !pip install PyMuPDF Pillow matplotlib numpy scipy gradio pandas

import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
from scipy import ndimage
import gradio as gr
import pandas as pd
from datetime import datetime
import uuid
from pathlib import Path
import shutil
import zipfile


class PDFCropPreservica:
    """PDF processor that saves images + metadata in Preservica-ready structure"""

    def __init__(self):
        self.current_image = None
        self.original_processed = None
        self.current_pdf_name = None
        self.current_page = 0
        self.click_count = 0
        self.click_coords = []
        self.last_crop = None

        self.pending_image = None
        self.pending_type = None
        self.crop_counter = 0
        self.save_counter = 0

        # Create output structure
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_folder = Path(f"preservica_export_{self.session_id}")
        self.images_folder = self.output_folder / "images"
        self.images_folder.mkdir(parents=True, exist_ok=True)

        self.dc_fields = ["file_or_folder_name", "Title", "Creator", "Subject", "Description", "Publisher",
                          "Contributor", "Date", "Type", "Format", "Identifier", "Source", "Language",
                          "Relation", "Coverage", "Rights"]
        self.history = {
            "Creator": [],
            "Subject": [],
            "Publisher": [],
            "Contributor": [],
            "Type": ["Text", "Image", "Audio", "Video", "Dataset"],
            "Format": ["PDF", "JPG", "PNG", "TIF", "MP3", "MP4"],
            "Language": ["English", "Spanish", "French", "German", "Italian"],
            "Rights": ["Access for educational and research purposes only.", "Public Domain", "Copyright protected"]
        }

        self.dc_data = {}
        self.dc_csv = self.output_folder / "dublin_core_metadata.csv"
        pd.DataFrame(columns=self.dc_fields).to_csv(self.dc_csv, index=False)

    def preprocess_image(self, img, contrast_level, sharpen_level, denoise_size,
                         deskew, use_binarize, bin_threshold, invert_level, brightness, dpi):
        processed_img = img.copy()
        if contrast_level > 0:
            processed_img = ImageEnhance.Contrast(processed_img).enhance(contrast_level)
        if sharpen_level > 0:
            for _ in range(int(sharpen_level)):
                processed_img = processed_img.filter(ImageFilter.SHARPEN)
        if denoise_size > 1:
            processed_img = processed_img.filter(ImageFilter.MedianFilter(size=int(denoise_size)))
        if brightness != 0:
            factor = 1.0 + (brightness / 200.0)
            processed_img = ImageEnhance.Brightness(processed_img).enhance(factor)
        if use_binarize:
            processed_img = processed_img.convert("L").point(lambda p: 255 if p > bin_threshold else 0).convert("RGB")
        if deskew:
            processed_img = self.deskew_image(processed_img)
        if invert_level > 0:
            inv = ImageOps.invert(processed_img.convert("RGB"))
            processed_img = Image.blend(processed_img, inv, invert_level)
        return processed_img

    def deskew_image(self, img):
        try:
            gray = img.convert('L')
            arr = np.array(gray)
            edges = arr > 128
            angles = np.linspace(-5, 5, 50)
            scores = []
            for angle in angles:
                rotated = ndimage.rotate(edges, angle, reshape=False, order=0)
                profile = np.sum(rotated, axis=1)
                scores.append(np.sum(np.diff(profile) ** 2))
            best_angle = angles[np.argmax(scores)]
            return img.rotate(best_angle, expand=True, fillcolor='white')
        except:
            return img

    def process_pdf_page(self, pdf_file, page_num, contrast_level=1.0, sharpen_level=0, denoise_size=1,
                         deskew=False, use_binarize=False, bin_threshold=128,
                         invert_level=0.0, brightness=0, dpi=150):
        if pdf_file is None:
            return self.current_image, None, "⚠️ Upload a PDF first", "", self.get_status()

        self.current_pdf = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        self.current_pdf_name = Path(self.current_pdf).stem
        self.current_page = page_num
        self.crop_counter = 0

        try:
            doc = fitz.open(self.current_pdf)
            if page_num >= len(doc):
                doc.close()
                return self.current_image, None, f"⚠️ Page {page_num + 1} not found", "", self.get_status()

            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            processed_img = self.preprocess_image(img, contrast_level, sharpen_level, denoise_size,
                                                  deskew, use_binarize, bin_threshold, invert_level, brightness, dpi)
            doc.close()

            self.current_image = processed_img
            self.original_processed = processed_img.copy()
            self.click_count = 0
            self.click_coords = []
            self.pending_image = None

            width, height = processed_img.size
            return processed_img, None, f"✅ Page {page_num + 1} loaded | {width}×{height}px", "", self.get_status()
        except Exception as e:
            return self.current_image, None, f"❌ Error: {str(e)}", "", self.get_status()

    def save_full_page(self):
        """Stage full page for metadata entry"""
        print(f"DEBUG: original_processed is None? {self.original_processed is None}")
        print(f"DEBUG: pending_image is None? {self.pending_image is None}")
        if self.original_processed is None:
            return None, "⚠️ Process a page first", "", self.get_status()

        self.pending_image = self.original_processed.copy()
        self.pending_type = "fullpage"
        suggested_filename = f"{self.current_pdf_name}_page{self.current_page + 1}.png"

        return self.pending_image, f"✅ Full page staged!\n👉 Fill metadata and click 'Confirm & Save'", suggested_filename, self.get_status()

    def handle_click(self, evt: gr.SelectData):
        """Handle crop selection"""
        if self.current_image is None:
            return self.current_image, None, "⚠️ Process a page first", "", self.get_status()

        x, y = evt.index

        if self.click_count == 0:
            self.click_coords = [(x, y)]
            self.click_count = 1
            preview = self.current_image.copy()
            draw = ImageDraw.Draw(preview)
            size = 30
            draw.line([x - size, y, x + size, y], fill='lime', width=6)
            draw.line([x, y - size, x, y + size], fill='lime', width=6)
            draw.ellipse([x - 20, y - 20, x + 20, y + 20], outline='lime', width=6)
            return preview, None, f"📍 First corner: ({x}, {y})\n👆 Click second corner", "", self.get_status()

        elif self.click_count == 1:
            self.click_coords.append((x, y))
            x1, y1 = self.click_coords[0]
            x2, y2 = self.click_coords[1]
            left, top = min(x1, x2), min(y1, y2)
            right, bottom = max(x1, x2), max(y1, y2)

            if right - left < 10 or bottom - top < 10:
                self.click_count = 0
                self.click_coords = []
                return self.current_image, None, "❌ Too small, try again", "", self.get_status()

            preview = self.current_image.copy()
            draw = ImageDraw.Draw(preview, 'RGBA')
            w, h = preview.size

            if top > 0:
                draw.rectangle([0, 0, w, top], fill=(0, 0, 0, 180))
            if bottom < h:
                draw.rectangle([0, bottom, w, h], fill=(0, 0, 0, 180))
            if left > 0:
                draw.rectangle([0, top, left, bottom], fill=(0, 0, 0, 180))
            if right < w:
                draw.rectangle([right, top, w, bottom], fill=(0, 0, 0, 180))

            draw.rectangle([left - 3, top - 3, right + 3, bottom + 3], outline='yellow', width=10)
            draw.rectangle([left, top, right, bottom], outline='lime', width=6)

            cropped = self.current_image.crop((left, top, right, bottom))
            self.last_crop = (left, top, right, bottom)
            self.pending_image = cropped
            self.pending_type = "crop"
            self.click_count = 0
            self.crop_counter += 1

            suggested_filename = f"{self.current_pdf_name}_page{self.current_page + 1}_crop{self.crop_counter}.png"
            return preview, cropped, f"✅ Crop staged: ({left},{top})→({right},{bottom})\n👉 Fill metadata and confirm", suggested_filename, self.get_status()

        return self.current_image, None, "", "", self.get_status()

    def confirm_save_metadata(self, file_or_folder_name, Title, Creator, Subject, Description, Publisher,
                              Contributor, Date, Type, Format, Identifier, Source, Language,
                              Relation, Coverage, Rights):
        """Save image to disk AND metadata to CSV"""
        if self.pending_image is None:
            return None, "⚠️ Nothing to save! Save full page or crop first.", "", self.get_status()

        if not file_or_folder_name:
            return self.pending_image, "❌ Filename is required!", file_or_folder_name, self.get_status()

        if file_or_folder_name in self.dc_data:
            return self.pending_image, f"❌ '{file_or_folder_name}' already exists!\nChange the filename.", file_or_folder_name, self.get_status()

        # Save the actual image file
        image_path = self.images_folder / file_or_folder_name
        try:
            self.pending_image.save(image_path)
        except Exception as e:
            return self.pending_image, f"❌ Failed to save image: {str(e)}", file_or_folder_name, self.get_status()

        # Save metadata to CSV
        dc_record = {
            "file_or_folder_name": file_or_folder_name,
            "Title": Title if Title else Path(file_or_folder_name).stem,
            "Creator": Creator or "",
            "Subject": Subject or "",
            "Description": Description or "",
            "Publisher": Publisher or "",
            "Contributor": Contributor or "",
            "Date": Date or "",
            "Type": Type or "",
            "Format": Format or "",
            "Identifier": Identifier or "",
            "Source": Source or "",
            "Language": Language or "",
            "Relation": Relation or "",
            "Coverage": Coverage or "",
            "Rights": Rights or ""
        }

        self.dc_data[file_or_folder_name] = dc_record
        df = pd.DataFrame([dc_record])
        df.to_csv(self.dc_csv, mode='a', header=False, index=False)
        self.save_counter += 1

        print(f"✅ SAVED: {file_or_folder_name}")
        print(f"📁 Image: {image_path}")
        print(f"📄 CSV: {self.dc_csv}")

        self.pending_image = None
        self.pending_type = None

        return None, f"✅ SUCCESS! Image and metadata saved\n💾 {file_or_folder_name}\n📁 {self.output_folder}\n\n🎯 Ready for next item", "", self.get_status()

    def reset_clicks(self):
        self.click_count = 0
        self.click_coords = []
        return self.original_processed or None, None, "🔄 Clicks reset", "", self.get_status()

    def reapply_last_crop(self):
        if not self.last_crop or not self.original_processed:
            return self.original_processed, None, "⚠️ No crop to reapply", "", self.get_status()

        left, top, right, bottom = self.last_crop
        preview = self.original_processed.copy()
        draw = ImageDraw.Draw(preview, 'RGBA')
        w, h = preview.size

        if top > 0:
            draw.rectangle([0, 0, w, top], fill=(0, 0, 0, 180))
        if bottom < h:
            draw.rectangle([0, bottom, w, h], fill=(0, 0, 0, 180))
        if left > 0:
            draw.rectangle([0, top, left, bottom], fill=(0, 0, 0, 180))
        if right < w:
            draw.rectangle([right, top, w, bottom], fill=(0, 0, 0, 180))

        draw.rectangle([left - 3, top - 3, right + 3, bottom + 3], outline='yellow', width=10)
        draw.rectangle([left, top, right, bottom], outline='lime', width=6)

        cropped = self.original_processed.crop((left, top, right, bottom))
        self.pending_image = cropped
        self.crop_counter += 1
        suggested = f"{self.current_pdf_name}_page{self.current_page + 1}_crop{self.crop_counter}.png"
        return preview, cropped, f"✅ Crop reapplied\n👉 Fill metadata and confirm", suggested, self.get_status()

    def get_status(self):
        """Return current session status"""
        return f"📊 Session: {self.save_counter} items saved\n📁 Output: {self.output_folder}"

    def create_preservica_package(self):
        """Create a ZIP file ready for Preservica upload"""
        if self.save_counter == 0:
            return None, "⚠️ No items saved yet. Save at least one image first."

        zip_path = Path(f"{self.output_folder}.zip")

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add CSV
                zipf.write(self.dc_csv, self.dc_csv.name)

                # Add all images
                for img_file in self.images_folder.glob("*"):
                    if img_file.is_file():
                        zipf.write(img_file, f"images/{img_file.name}")

            return str(
                zip_path), f"✅ Preservica package created!\n📦 {zip_path}\n📄 {self.save_counter} images + metadata CSV"
        except Exception as e:
            return None, f"❌ Failed to create package: {str(e)}"


def create_gradio_interface():
    processor = PDFCropPreservica()

    with gr.Blocks(title="PDF to Preservica") as demo:
        gr.Markdown("# 📄 PDF Crop & Metadata → Preservica Export")
        gr.Markdown("**Saves images + Dublin Core CSV in organized folder structure**")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📁 STEP 1: Process Page")
                with gr.Row():
                    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                    page_num = gr.Slider(0, 100, value=0, step=1, label="Page")
                process_btn = gr.Button("🔄 Process Page", variant="primary", size="lg")

                with gr.Accordion("⚙️ Preprocessing", open=False):
                    with gr.Row():
                        contrast_level = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Contrast")
                        brightness = gr.Slider(-100, 100, value=0, label="Brightness")
                    with gr.Row():
                        sharpen_level = gr.Slider(0, 5, value=0, step=1, label="Sharpen")
                        denoise_size = gr.Slider(1, 7, value=1, step=2, label="Denoise")
                    deskew = gr.Checkbox(label="Deskew")
                    use_binarize = gr.Checkbox(label="Binarize")
                    bin_threshold = gr.Slider(0, 255, value=128, label="Threshold")
                    invert_level = gr.Slider(0.0, 1.0, value=0.0, label="Invert")
                    dpi = gr.Slider(72, 600, value=150, step=50, label="DPI")

                gr.Markdown("---")
                gr.Markdown("### ✂️ STEP 2: Save or Crop")
                save_full_btn = gr.Button("💾 Save Full Page", variant="secondary", size="lg")
                gr.Markdown("*OR click two corners on preview to crop*")
                with gr.Row():
                    reset_btn = gr.Button("↺ Reset Clicks", size="sm")
                    reapply_btn = gr.Button("🔄 Reapply Crop", size="sm")

                gr.Markdown("---")
                gr.Markdown("### 📝 STEP 3: Fill Metadata")
                dc_file = gr.Textbox(label="📁 Filename (Required)", placeholder="Auto-generated")
                dc_title = gr.Textbox(label="Title")
                with gr.Row():
                    dc_creator = gr.Dropdown(label="Creator", choices=processor.history["Creator"],
                                             allow_custom_value=True)
                    dc_publisher = gr.Dropdown(label="Publisher", choices=processor.history["Publisher"],
                                               allow_custom_value=True)
                dc_subject = gr.Dropdown(label="Subject", choices=processor.history["Subject"], allow_custom_value=True,
                                         multiselect=True)
                dc_description = gr.Textbox(label="Description", lines=2)
                with gr.Row():
                    dc_date = gr.Textbox(label="Date")
                    dc_language = gr.Dropdown(label="Language", choices=processor.history["Language"], value="English",
                                              allow_custom_value=True)

                with gr.Accordion("📋 More Fields", open=False):
                    dc_contributor = gr.Dropdown(label="Contributor", choices=processor.history["Contributor"],
                                                 allow_custom_value=True, multiselect=True)
                    with gr.Row():
                        dc_type = gr.Dropdown(label="Type", choices=processor.history["Type"], value="Text",
                                              allow_custom_value=True)
                        dc_format = gr.Dropdown(label="Format", choices=processor.history["Format"], value="PNG",
                                                allow_custom_value=True)
                    dc_identifier = gr.Textbox(label="Identifier")
                    dc_source = gr.Textbox(label="Source")
                    dc_relation = gr.Textbox(label="Relation")
                    dc_coverage = gr.Textbox(label="Coverage")
                    dc_rights = gr.Dropdown(label="Rights", choices=processor.history["Rights"],
                                            value="Access for educational and research purposes only.",
                                            allow_custom_value=True)

                gr.Markdown("---")
                gr.Markdown("### ✅ STEP 4: Confirm & Save")
                confirm_btn = gr.Button("💾 CONFIRM & SAVE (Image + Metadata)", variant="primary", size="lg")

                session_status = gr.Markdown(processor.get_status())
                status_text = gr.Textbox(label="Status", interactive=False, lines=4)

                gr.Markdown("---")
                gr.Markdown("### 📦 STEP 5: Export for Preservica")
                export_btn = gr.Button("📦 Create Preservica Package (ZIP)", variant="primary", size="lg")
                package_file = gr.File(label="Download Package")
                export_status = gr.Textbox(label="Export Status", interactive=False, lines=2)

            with gr.Column(scale=3):
                gr.Markdown("### 👁️ Preview (Click to Crop)")
                preview_image = gr.Image(type="pil", label="Click two corners", height=500)
                gr.Markdown("### 📤 Staged Image (Awaiting Confirmation)")
                output_image = gr.Image(type="pil", label="Fill metadata and confirm", height=400)

        # Event handlers
        process_btn.click(
            fn=processor.process_pdf_page,
            inputs=[pdf_input, page_num, contrast_level, sharpen_level, denoise_size,
                    deskew, use_binarize, bin_threshold, invert_level, brightness, dpi],
            outputs=[preview_image, output_image, status_text, dc_file, session_status]
        )

        save_full_btn.click(
            fn=processor.save_full_page,
            outputs=[output_image, status_text, dc_file, session_status]
        )

        preview_image.select(
            fn=processor.handle_click,
            outputs=[preview_image, output_image, status_text, dc_file, session_status]
        )

        reset_btn.click(
            fn=processor.reset_clicks,
            outputs=[preview_image, output_image, status_text, dc_file, session_status]
        )

        reapply_btn.click(
            fn=processor.reapply_last_crop,
            outputs=[preview_image, output_image, status_text, dc_file, session_status]
        )

        confirm_btn.click(
            fn=processor.confirm_save_metadata,
            inputs=[dc_file, dc_title, dc_creator, dc_subject, dc_description, dc_publisher,
                    dc_contributor, dc_date, dc_type, dc_format, dc_identifier, dc_source,
                    dc_language, dc_relation, dc_coverage, dc_rights],
            outputs=[output_image, status_text, dc_file, session_status]
        )

        export_btn.click(
            fn=processor.create_preservica_package,
            outputs=[package_file, export_status]
        )

        gr.Markdown("""
        ---
        ## 📖 Instructions:

        **STEP 1:** Upload PDF → Select page → Process

        **STEP 2:** Save full page OR click two corners to crop

        **STEP 3:** Fill metadata fields

        **STEP 4:** Click "CONFIRM & SAVE" → **Image saved to disk + CSV updated**

        **REPEAT 2-4** for all images you need

        **STEP 5:** Click "Create Preservica Package" → Downloads ZIP file

        ---

        ### 📁 Output Structure:
        ```
        preservica_export_YYYYMMDD_HHMMSS/
        ├── images/
        │   ├── document_page1.png
        │   ├── document_page1_crop1.png
        │   └── document_page2_crop1.png
        └── dublin_core_metadata.csv
        ```

        ### 💡 Tips:
        - Images are saved immediately when you confirm
        - CSV updates with each confirmation
        - ZIP package contains everything for Preservica upload
        - Session folder name includes timestamp
        """)

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
