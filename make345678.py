import os
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

def generate_exact_aruco_pdf():
    pdf_filename = "aruco_30mm_single_page.pdf"
    
    # 1. 呼叫 OpenCV 官方 DICT_4X4_50 字典
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    target_ids = [3, 4, 5, 6, 7, 8]
    temp_files = []
    
    # 2. 用 OpenCV 產出官方標準的高解析度 PNG 圖檔
    for m_id in target_ids:
        # 相容新舊版 OpenCV API
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(dictionary, m_id, 600)
        else:
            img = cv2.aruco.drawMarker(dictionary, m_id, 600)
            
        filename = f"temp_exact_marker_{m_id}.png"
        cv2.imwrite(filename, img)
        temp_files.append(filename)

    # 3. 建立 A4 PDF (210 mm x 297 mm)
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4  # 單位為點 (pt)
    
    # 標題與說明文字
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_width / 2.0, page_height - 20 * mm, "ArUco Markers (DICT_4X4_50) - Side Trackers")
    
    c.setFont("Helvetica", 9)
    c.drawCentredString(page_width / 2.0, page_height - 26 * mm, "Print at 100% Scale / Actual Size | Exact Marker Size: 30 mm x 30 mm")

    # 4. 精確排版 2 欄 x 3 列 (共 6 個 Marker，嚴格單頁收納)
    marker_size = 30 * mm  # 嚴格設定實體邊長為 30 mm
    
    # 計算 X 軸位置 (左右兩欄)
    col_width = 80 * mm
    left_margin = (page_width - (2 * col_width)) / 2.0
    x_positions = [left_margin + 10 * mm, left_margin + col_width + 10 * mm]
    
    # 計算 Y 軸位置 (三列)
    y_positions = [
        page_height - 75 * mm,
        page_height - 135 * mm,
        page_height - 195 * mm
    ]

    for i, m_id in enumerate(target_ids):
        col = i % 2
        row = i // 2
        
        x = x_positions[col]
        y = y_positions[row]
        
        # 繪製灰色裁切虛線框 (外加 2mm 留白 margin)
        margin = 2 * mm
        c.setDash(2, 2)
        c.setLineWidth(0.5)
        c.rect(x - margin, y - margin, marker_size + 2 * margin, marker_size + 2 * margin)
        
        # 插入 OpenCV 生成的標準圖案
        c.drawImage(temp_files[i], x, y, width=marker_size, height=marker_size)
        
        # 標示 ID 文字
        c.setDash()  # 恢復實線
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x + marker_size + 4 * mm, y + (marker_size / 2.0) - 1.5 * mm, f"ID {m_id}")

    # 底部加上 50mm 實體校對尺，方便印出後驗證
    scale_y = 35 * mm
    scale_x_start = (page_width - 50 * mm) / 2.0
    c.setLineWidth(1)
    c.line(scale_x_start, scale_y, scale_x_start + 50 * mm, scale_y)
    c.line(scale_x_start, scale_y - 2 * mm, scale_x_start, scale_y + 4 * mm)
    c.line(scale_x_start + 50 * mm, scale_y - 2 * mm, scale_x_start + 50 * mm, scale_y + 4 * mm)
    c.setFont("Helvetica", 8)
    c.drawCentredString(page_width / 2.0, scale_y - 5 * mm, "Calibration Check Line: Exactly 50 mm")

    # 儲存檔案
    c.save()
    
    # 清除暫存圖片
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            
    print(f"✅ 成功生成 100% 官方標準二進位碼單頁 PDF：{pdf_filename}")

if __name__ == "__main__":
    generate_exact_aruco_pdf()