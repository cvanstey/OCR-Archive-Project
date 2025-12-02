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


class PDFCropSelectorWithDC:
    """PDF processor with click-to-crop + Dublin Core metadata"""

    def __init__(self):
        self.current_image = None
        self.original_processed = None
        self.current_pdf_name = None
        self.click_count = 0
        self.click_coords = []
        self.last_crop = None
        self.pending_metadata = {}
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
        self.session_id = str(uuid.uuid4())
        self.dc_csv = Path(f"dublin_core_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        # initialize CSV
        pd.DataFrame(columns=self.dc_fields).to_csv(self.dc_csv, index=False)

    # ----------------- Image Processing -----------------
    def preprocess_image(self, img,
                         contrast_level, sharpen_level, denoise_size,
                         deskew, use_binarize, bin_threshold,
                         invert_level, brightness, dpi):
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

    def process_pdf_page(self, pdf_file, page_num,
                         contrast_level=1.0, sharpen_level=0, denoise_size=1,
                         deskew=False, use_binarize=False, bin_threshold=128,
                         invert_level=0.0, brightness=0, dpi=150):
        if pdf_file is None:
            # Keep existing cropped image if there is one
            return self.current_image, self.last_cropped_result if hasattr(self,
                                                                           'last_cropped_result') else None, "Upload a PDF first", 0, 0, 0, 0

        self.current_pdf = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        self.current_page = page_num
        self.current_params = {
            "contrast_level": contrast_level, "sharpen_level": sharpen_level, "denoise_size": denoise_size,
            "deskew": deskew, "use_binarize": use_binarize, "bin_threshold": bin_threshold,
            "invert_level": invert_level, "brightness": brightness, "dpi": dpi
        }

        try:
            doc = fitz.open(self.current_pdf)
            if page_num >= len(doc):
                doc.close()
                # Keep existing cropped image
                return self.current_image, self.last_cropped_result if hasattr(self,
                                                                               'last_cropped_result') else None, f"Page {page_num + 1} not found. PDF has {len(doc)} pages.", 0, 0, 0, 0

            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            processed_img = self.preprocess_image(img, contrast_level, sharpen_level, denoise_size,
                                                  deskew, use_binarize, bin_threshold,
                                                  invert_level, brightness, dpi)
            doc.close()

            self.current_image = processed_img
            self.original_processed = processed_img.copy()
            self.click_count = 0
            self.click_coords = []
            # DON'T reset last_crop so it persists
            # DON'T clear last_cropped_result

            width, height = processed_img.size
            # Keep showing the last cropped result
            return processed_img, self.last_cropped_result if hasattr(self,
                                                                      'last_cropped_result') else None, f"✅ Page {page_num + 1} | Size: {width}×{height}px\n👆 Click two corners to crop", 0, 0, width, height

        except Exception as e:
            return self.current_image, self.last_cropped_result if hasattr(self,
                                                                           'last_cropped_result') else None, f"❌ Error: {str(e)}", 0, 0, 0, 0

    # ----------------- Cropping -----------------
    def handle_click(self, evt: gr.SelectData,
                     file_or_folder_name, Title, Creator, Subject, Description, Publisher,
                     Contributor, Date, Type, Format, Identifier, Source, Language,
                     Relation, Coverage, Rights):
        """
        Handle a click on the preview image to select crop corners.
        evt.index gives (x, y) coordinates.
        """

        if self.current_image is None:
            return self.current_image, None, "⚠️ No image loaded. Process a PDF page first.", 0, 0, 0, 0

        # Get click coordinates from Gradio SelectData
        x, y = evt.index

        print(f"Click detected at: ({x}, {y}), count: {self.click_count}")

        # ---- First corner ----
        if self.click_count == 0:
            self.click_coords = [(x, y)]
            self.click_count = 1

            # Draw first corner marker
            preview = self.current_image.copy()
            draw = ImageDraw.Draw(preview)
            size = 30
            draw.line([x - size, y, x + size, y], fill='lime', width=6)
            draw.line([x, y - size, x, y + size], fill='lime', width=6)
            draw.ellipse([x - 20, y - 20, x + 20, y + 20], outline='lime', width=6)

            return preview, None, f"📍 First corner: ({x}, {y})\n👆 Click second corner to complete crop", x, y, 0, 0

        # ---- Second corner ----
        elif self.click_count == 1:
            self.click_coords.append((x, y))
            x1, y1 = self.click_coords[0]
            x2, y2 = self.click_coords[1]

            # Calculate crop box
            left, top = min(x1, x2), min(y1, y2)
            right, bottom = max(x1, x2), max(y1, y2)

            # Ensure valid crop dimensions
            if right - left < 10 or bottom - top < 10:
                self.click_count = 0
                self.click_coords = []
                return self.current_image, None, "❌ Crop area too small. Try again.", 0, 0, 0, 0

            # Draw crop preview
            preview = self.current_image.copy()
            draw = ImageDraw.Draw(preview, 'RGBA')
            w, h = preview.size

            # Shade areas outside crop (semi-transparent black)
            if top > 0:
                draw.rectangle([0, 0, w, top], fill=(0, 0, 0, 180))
            if bottom < h:
                draw.rectangle([0, bottom, w, h], fill=(0, 0, 0, 180))
            if left > 0:
                draw.rectangle([0, top, left, bottom], fill=(0, 0, 0, 180))
            if right < w:
                draw.rectangle([right, top, w, bottom], fill=(0, 0, 0, 180))

            # Draw crop rectangle
            draw.rectangle([left - 3, top - 3, right + 3, bottom + 3], outline='yellow', width=10)
            draw.rectangle([left, top, right, bottom], outline='lime', width=6)

            # Draw corner markers
            size = 30
            for cx, cy in [(left, top), (right, top), (left, bottom), (right, bottom)]:
                draw.line([cx - size, cy, cx + size, cy], fill='lime', width=6)
                draw.line([cx, cy - size, cx, cy + size], fill='lime', width=6)

            # Actually crop the image
            cropped = self.current_image.crop((left, top, right, bottom))
            self.last_crop = (left, top, right, bottom)
            self.last_cropped_result = cropped  # SAVE the cropped image
            self.click_count = 0  # Reset for next crop

            # ---- Save Dublin Core metadata ----
            dc_record = {
                "file_or_folder_name": file_or_folder_name or "",
                "Title": Title or "",
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

            filename = f"crop_{self.current_page + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            # Set filename if not provided
            if not dc_record["file_or_folder_name"]:
                dc_record["file_or_folder_name"] = filename

            # Default Title = filename without extension
            if not dc_record["Title"]:
                dc_record["Title"] = Path(filename).stem

            self.dc_data[filename] = dc_record

            # Append to CSV
            df = pd.DataFrame([dc_record])
            df.to_csv(self.dc_csv, mode='a', header=False, index=False)

            print(f"Crop saved: {left},{top} -> {right},{bottom}")
            print(f"Metadata saved to: {self.dc_csv}")

            return preview, cropped, f"✅ Crop: ({left},{top}) → ({right},{bottom})\n💾 Saved to {self.dc_csv}", left, top, right, bottom

        return self.current_image, None, "⚠️ Unexpected state", 0, 0, 0, 0

    def reapply_last_crop(self):
        """Reapply the last crop coordinates to the current image"""
        if self.last_crop is None:
            return self.original_processed, None, "⚠️ No crop coordinates saved"
        if self.original_processed is None:
            return None, None, "⚠️ No image loaded"

        left, top, right, bottom = self.last_crop

        # Draw preview
        preview = self.original_processed.copy()
        draw = ImageDraw.Draw(preview, 'RGBA')
        w, h = preview.size

        # Shade outside areas
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

        # Crop
        cropped = self.original_processed.crop((left, top, right, bottom))

        return preview, cropped, f"✅ Crop reapplied: {cropped.size[0]}×{cropped.size[1]}px"

    def reset_clicks(self):
        """Reset click counter and return to original processed image"""
        self.click_count = 0
        self.click_coords = []
        img = self.original_processed if self.original_processed else None
        return img, None, "🔄 Clicks reset. Ready for new crop selection."

    def save_full_page(self):
        """Save the full preprocessed page without cropping"""
        if self.original_processed is None:
            return None, "⚠️ No image to save. Process a page first."

        # Use pending metadata or create empty record
        if self.pending_metadata:
            dc_record = self.pending_metadata.copy()
        else:
            dc_record = {field: "" for field in self.dc_fields}

        # Generate filename with PDF name
        filename = f"{self.current_pdf_name}_fullpage{self.current_page + 1}_{datetime.now().strftime('%H%M%S')}.png"

        # Set filename if not provided
        if not dc_record["file_or_folder_name"]:
            dc_record["file_or_folder_name"] = filename

        # Default Title = PDF name + page
        if not dc_record["Title"]:
            dc_record["Title"] = f"{self.current_pdf_name} - Page {self.current_page + 1}"

        self.dc_data[filename] = dc_record

        # Append to CSV
        df = pd.DataFrame([dc_record])
        df.to_csv(self.dc_csv, mode='a', header=False, index=False)

        # Clear pending metadata after use
        self.pending_metadata = {}

        print(f"Full page saved: {filename}")
        print(f"Metadata saved to: {self.dc_csv}")

        return self.original_processed, f"✅ Full page saved with metadata!\n💾 {filename} added to {self.dc_csv}"

    def get_csv_path(self):
        """Return the CSV file path for download"""
        if self.dc_csv.exists():
            return str(self.dc_csv)
        return None

# ----------------- Gradio Interface -----------------
def create_gradio_interface():
    processor = PDFCropSelectorWithDC()

    with gr.Blocks(title="PDF Crop + DC Metadata") as demo:
        gr.Markdown("# 📄 PDF Crop & Dublin Core Metadata Tool")
        gr.Markdown("**Upload a PDF, preprocess it, then click two corners to crop. Metadata is saved automatically.**")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 File & Page")
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                page_num = gr.Slider(0, 100, value=0, step=1, label="Page Number (0 = first page)")

                with gr.Accordion("⚙️ Preprocessing Options", open=False):
                    contrast_level = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Contrast")
                    sharpen_level = gr.Slider(0, 5, value=0, step=1, label="Sharpen")
                    denoise_size = gr.Slider(1, 7, value=1, step=2, label="Denoise Size")
                    deskew = gr.Checkbox(label="Deskew (Straighten)")
                    use_binarize = gr.Checkbox(label="Binarize (Black & White)")
                    bin_threshold = gr.Slider(0, 255, value=128, label="Binarize Threshold")
                    invert_level = gr.Slider(0.0, 1.0, step=0.1, value=0.0, label="Invert Level")
                    brightness = gr.Slider(-100, 100, value=0, label="Brightness")
                    dpi = gr.Slider(72, 600, value=150, step=50, label="DPI")

                process_btn = gr.Button("🔄 Process Page", variant="primary", size="lg")

                gr.Markdown("### ✂️ Crop Coordinates (Auto-filled)")
                with gr.Row():
                    crop_left = gr.Number(value=0, label="Left", interactive=False)
                    crop_top = gr.Number(value=0, label="Top", interactive=False)
                with gr.Row():
                    crop_right = gr.Number(value=0, label="Right", interactive=False)
                    crop_bottom = gr.Number(value=0, label="Bottom", interactive=False)

                with gr.Row():
                    reset_btn = gr.Button("↺ Reset Clicks", size="sm")
                    reapply_btn = gr.Button("🔄 Reapply Last Crop", variant="secondary", size="sm")
                save_full_btn = gr.Button("💾 Save Full Page (No Crop)", variant="secondary",
                                          size="lg")

                gr.Markdown("### 💾 Download Metadata")
                csv_file = gr.File(label="Dublin Core CSV", interactive=False)
                download_csv_btn = gr.Button("📥 Get Current CSV", size="sm")

                with gr.Accordion("📝 Dublin Core Metadata", open=True):
                    gr.Markdown("**Fill out fields, then click 'Save Metadata' before cropping**")
                    dc_file = gr.Textbox(label="file_or_folder_name", placeholder="Auto-generated", interactive=True)
                    dc_title = gr.Textbox(label="Title", placeholder="Auto-generated from filename", interactive=True)
                    dc_creator = gr.Dropdown(label="Creator", choices=processor.history["Creator"],
                                             allow_custom_value=True)
                    dc_subject = gr.Dropdown(label="Subject (can select multiple)",
                                             choices=processor.history["Subject"], allow_custom_value=True,
                                             multiselect=True)
                    dc_description = gr.Textbox(label="Description")
                    dc_publisher = gr.Dropdown(label="Publisher", choices=processor.history["Publisher"],
                                               allow_custom_value=True)
                    dc_contributor = gr.Dropdown(label="Contributor (can select multiple)",
                                                 choices=processor.history["Contributor"], allow_custom_value=True,
                                                 multiselect=True)
                    dc_date = gr.Textbox(label="Date", placeholder="YYYY-MM-DD")
                    dc_type = gr.Dropdown(label="Type", choices=processor.history["Type"], value="Text",
                                          allow_custom_value=True)
                    dc_format = gr.Dropdown(label="Format", choices=processor.history["Format"], value="PDF",
                                            allow_custom_value=True)
                    dc_identifier = gr.Textbox(label="Identifier")
                    dc_source = gr.Textbox(label="Source")
                    dc_language = gr.Dropdown(label="Language", choices=processor.history["Language"], value="English",
                                              allow_custom_value=True)
                    dc_relation = gr.Textbox(label="Relation")
                    dc_coverage = gr.Textbox(label="Coverage")
                    dc_rights = gr.Dropdown(label="Rights", choices=processor.history["Rights"],
                                            value="Access for educational and research purposes only.",
                                            allow_custom_value=True)

                    save_metadata_btn = gr.Button("💾 Save Metadata", variant="primary", size="lg")
                    metadata_status = gr.Textbox(label="Metadata Status", interactive=False, lines=1)

        with gr.Row():
            with gr.Column(scale=1):
                preview_image = gr.Image(
                    type="pil",
                    label="👆 Click Two Corners to Crop",
                    height=700,
                    interactive=False
                )
            with gr.Column(scale=1):
                output_image = gr.Image(
                    type="pil",
                    label="✅ Cropped Result",
                    height=700
                )

                status_text = gr.Textbox(label="📊 Status", interactive=False, lines=2)

        # Process PDF button
        process_btn.click(
            fn=processor.process_pdf_page,
            inputs=[pdf_input, page_num,
                    contrast_level, sharpen_level, denoise_size,
                    deskew, use_binarize, bin_threshold,
                    invert_level, brightness, dpi],
            outputs=[preview_image, output_image, status_text, crop_left, crop_top, crop_right, crop_bottom]
        )

        # Handle crop clicks - FIXED: Use all DC textboxes as inputs
        preview_image.select(
            fn=processor.handle_click,
            inputs=[dc_file, dc_title, dc_creator, dc_subject, dc_description, dc_publisher,
                    dc_contributor, dc_date, dc_type, dc_format, dc_identifier, dc_source,
                    dc_language, dc_relation, dc_coverage, dc_rights],
            outputs=[preview_image, output_image, status_text, crop_left, crop_top, crop_right, crop_bottom]
        )

        # Reapply crop
        reapply_btn.click(
            fn=processor.reapply_last_crop,
            outputs=[preview_image, output_image, status_text]
        )

        # Reset clicks
        reset_btn.click(
            fn=processor.reset_clicks,
            outputs=[preview_image, output_image, status_text]
        )

        # Save full page without cropping
        save_full_btn.click(
            fn=processor.save_full_page,
            outputs=[output_image, status_text]
        )

        # Download CSV button
        download_csv_btn.click(
            fn=processor.get_csv_path,
            outputs=csv_file
        )

        gr.Markdown("""
        ### 📌 Instructions:
        1. **Upload PDF** and select page number
        2. **Process Page** to apply preprocessing
        3. **Click two corners** on the preview image to select crop area
        4. **Fill metadata** (optional - some fields auto-generate)
        5. **Download** cropped image using download button
        6. Metadata is auto-saved to CSV file

        ### 💡 Tips:
        - Click "Reset Clicks" if you make a mistake
        - Use "Reapply Last Crop" to apply same crop to newly processed page
        - CSV file contains all metadata for each crop
        """)

    return demo


# ----------------- Launch -----------------
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)