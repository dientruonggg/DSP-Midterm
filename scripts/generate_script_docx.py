import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    for old_shd in tcPr.findall(qn("w:shd")):
        tcPr.remove(old_shd)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D0D5DD", sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_docx(filename):
    doc = docx.Document()
    
    # Page setup - A4
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    normal_style.paragraph_format.line_spacing = 1.2
    normal_style.paragraph_format.space_after = Pt(4)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    run_sub = p_title.add_run("BÁO CÁO GIỮA KỲ XỬ LÝ TÍN HIỆU SỐ (DSP 2026) — NHÓM 08\n")
    run_sub.font.size = Pt(10)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)
    
    run_main = p_title.add_run("KỊCH BẢN THUYẾT TRÌNH VOICE ACTIVITY DETECTION (VAD)\n")
    run_main.font.size = Pt(18)
    run_main.font.bold = True
    run_main.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A) # Deep Blue
    
    run_range = p_title.add_run("TỪ SLIDE 11 ĐẾN SLIDE 26 (THUẬT TOÁN 3, THỰC NGHIỆM, CRITIQUE & PHẢN BIỆN Q&A)")
    run_range.font.size = Pt(11)
    run_range.font.bold = True
    run_range.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Guide callout box
    table_guide = doc.add_table(rows=1, cols=1)
    table_guide.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table_guide.cell(0, 0)
    set_cell_background(cell, "F0F9FF")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    p_g = cell.paragraphs[0]
    p_g.paragraph_format.space_after = Pt(2)
    r = p_g.add_run("🎙️ HƯỚNG DẪN KỸ NĂNG NÓI TO, RÕ RÀNG VÀ TỰ TIN:\n")
    r.font.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(0x03, 0x69, 0xA1)
    
    tips = [
        "• Ký hiệu // : Điểm ngắt nghỉ ngắn (0.5 – 1 giây) để lấy hơi bằng bụng (hít sâu phình bụng), giúp giọng to, dày và không hụt hơi.",
        "• Chữ IN ĐẬM : Từ khóa kỹ thuật trọng tâm cần nhấn mạnh dằn giọng dứt khoát.",
        "• Ký hiệu [Hành động] : Chỉ dẫn cử chỉ tay, mắt quét và vị trí chỉ vào màn hình slide.",
        "• Mục 💡 Bản chất kỹ thuật : Diễn giải trực quan cốt lõi để tự tin trả lời vấn đáp của Thầy/Cô."
    ]
    for t in tips:
        p_t = cell.add_paragraph()
        p_t.paragraph_format.space_after = Pt(2)
        rt = p_t.add_run(t)
        rt.font.size = Pt(9.5)
        rt.font.color.rgb = RGBColor(0x0C, 0x4A, 0x6E)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    def add_slide_section(num, title, subtitle, duration, goal, actions, speech_text, tech_insight=None, table_data=None):
        # Heading
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(2)
        h.paragraph_format.keep_with_next = True
        
        rh_num = h.add_run(f"SLIDE {num}: {title.upper()}\n")
        rh_num.font.size = Pt(13)
        rh_num.font.bold = True
        rh_num.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)
        
        rh_sub = h.add_run(f"({subtitle}) — Thời lượng dự kiến: {duration}")
        rh_sub.font.size = Pt(10)
        rh_sub.font.italic = True
        rh_sub.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

        # Meta info
        pm = doc.add_paragraph()
        pm.paragraph_format.space_after = Pt(4)
        rm_g = pm.add_run("🎯 Mục tiêu: ")
        rm_g.font.bold = True
        rm_g.font.size = Pt(10)
        rm_g.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
        pm.add_run(goal).font.size = Pt(10)

        # Action cue
        pa = doc.add_paragraph()
        pa.paragraph_format.space_after = Pt(6)
        ra_lbl = pa.add_run("🎬 Hành động & Cử chỉ: ")
        ra_lbl.font.bold = True
        ra_lbl.font.size = Pt(10)
        ra_lbl.font.color.rgb = RGBColor(0x92, 0x40, 0x0E)
        ra_val = pa.add_run(actions)
        ra_val.font.size = Pt(10)
        ra_val.font.italic = True
        ra_val.font.color.rgb = RGBColor(0x78, 0x35, 0x0F)

        # Speech Box
        tbl_speech = doc.add_table(rows=1, cols=1)
        tbl_speech.alignment = WD_TABLE_ALIGNMENT.CENTER
        c_speech = tbl_speech.cell(0, 0)
        set_cell_background(c_speech, "F8FAFC")
        set_cell_margins(c_speech, top=120, bottom=120, left=180, right=180)
        
        ps = c_speech.paragraphs[0]
        ps.paragraph_format.space_after = Pt(4)
        rs_lbl = ps.add_run("🗣️ LỜI THOẠI TRỰC TIẾP (NÓI TO, DÕNG DẠC, ĐÚNG NHỊP):\n")
        rs_lbl.font.bold = True
        rs_lbl.font.size = Pt(10)
        rs_lbl.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

        lines = speech_text.strip().split("\n\n")
        for idx, line in enumerate(lines):
            p_line = c_speech.add_paragraph() if idx > 0 else ps
            p_line.paragraph_format.space_after = Pt(4)
            p_line.paragraph_format.line_spacing = 1.25
            
            # Simple bold parsing for **text**
            parts = line.split("**")
            is_bold = False
            for part in parts:
                if part:
                    r_p = p_line.add_run(part)
                    r_p.font.size = Pt(10.5)
                    r_p.font.name = "Calibri"
                    if is_bold:
                        r_p.font.bold = True
                        r_p.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                    else:
                        r_p.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
                is_bold = not is_bold

        # Optional Table
        if table_data:
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            headers, rows = table_data
            t = doc.add_table(rows=len(rows)+1, cols=len(headers))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            set_table_borders(t)
            
            # Header row
            for j, h_text in enumerate(headers):
                cell_h = t.cell(0, j)
                set_cell_background(cell_h, "E2E8F0")
                set_cell_margins(cell_h, top=80, bottom=80, left=100, right=100)
                ph = cell_h.paragraphs[0]
                ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                rh = ph.add_run(h_text)
                rh.font.bold = True
                rh.font.size = Pt(9.5)
                rh.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
            
            # Content rows
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    cell_r = t.cell(i+1, j)
                    if i % 2 == 1:
                        set_cell_background(cell_r, "F8FAFC")
                    set_cell_margins(cell_r, top=70, bottom=70, left=100, right=100)
                    pr = cell_r.paragraphs[0]
                    pr.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    rr = pr.add_run(val)
                    rr.font.size = Pt(9)
                    if j == 0:
                        rr.font.bold = True

        # Tech insight
        if tech_insight:
            pi = doc.add_paragraph()
            pi.paragraph_format.space_before = Pt(4)
            pi.paragraph_format.space_after = Pt(8)
            ri_lbl = pi.add_run("💡 Bản chất kỹ thuật (Deep-dive): ")
            ri_lbl.font.bold = True
            ri_lbl.font.size = Pt(9.5)
            ri_lbl.font.color.rgb = RGBColor(0x04, 0x78, 0x57) # Green
            ri_val = pi.add_run(tech_insight)
            ri_val.font.size = Pt(9.5)
            ri_val.font.color.rgb = RGBColor(0x06, 0x5F, 0x46)

    # Slide 11
    add_slide_section(
        11, "STUDENT 3 · ALGORITHM 03", "Parametric Gaussian Bayes Model — Trương Bùi Điền", "35 – 45 giây",
        "Chuyển giao quyền trình bày, định vị phương pháp tiếp cận xác suất thống kê cổ điển.",
        "Bước lên nửa bước, mắt nhìn thẳng Hội đồng, mỉm cười tự tin, tay mở hướng về slide.",
        "Kính thưa Thầy và các bạn! // Tiếp nối hai thuật toán tìm kiếm nhị phân và lược đồ thích nghi Histogram của Toàn và Hoàng, // em là **Trương Bùi Điền**, // sau đây xin đại diện nhóm trình bày **Thuật toán 03: Mô hình phân lớp Bayes tham số Gaussian (Parametric Gaussian Bayes Model)**. //\n\n"
        "Nếu như hai thuật toán trước mang tính **kinh nghiệm thực nghiệm**, // thì Thuật toán 3 tiếp cận bài toán VAD dưới lăng kính **Nhận dạng mẫu thống kê cổ điển (Statistical Pattern Recognition)**. // Ý tưởng cốt lõi là coi năng lượng của **khoảng lặng** và **tiếng nói** là hai phân phối Gaussian độc lập. // Từ tập huấn luyện, ta ước lượng tham số $\\mu$ và $\\sigma$, // rồi **giải phương trình bậc hai giải tích** để tìm ngưỡng quyết định tối ưu Bayes. //\n\n"
        "Ưu điểm lớn nhất là: // Ngưỡng phân tách hoàn toàn **có cơ sở toán học xác suất vững chắc**, // không phụ thuộc vào việc mò mẫm tham số! //",
        "Mô hình giả định biến ngẫu nhiên x (Short-Time Energy chuẩn hóa) tuân theo hai hàm mật độ xác suất chuẩn p(x|Sil) ~ N(mu_sil, sigma_sil^2) và p(x|Sp) ~ N(mu_sp, sigma_sp^2)."
    )

    # Slide 12
    add_slide_section(
        12, "ALG 3 · CORE METHOD", "From Empirical Statistics to Analytical Bayes Root", "60 – 75 giây",
        "Giải thích chi tiết 6 bước từ trích xuất STE đến giải nghiệm phương trình bậc 2.",
        "Tay quét ngang theo 6 khối từ trái sang phải, sau đó chỉ vào từng tham số bên dưới.",
        "Trên màn hình là **quy trình xử lý 6 bước khép kín** của mô hình Bayes: //\n\n"
        "1. **Bước 1**: Trích xuất năng lượng ngắn hạn chuẩn hóa STE trên **4 file huấn luyện**. //\n"
        "2. **Bước 2**: Dựa vào nhãn Praat Ground-Truth, tách các khung thành hai tập: **Khoảng lặng (Silence)** và **Tiếng nói (Speech)**. //\n"
        "3. **Bước 3**: Khớp hai hàm mật độ Gaussian: //\n"
        "   - Khoảng lặng có trung bình cực nhỏ: **mu_sil = 0.00039**, độ lệch rất hẹp: **sigma_sil = 0.00071**. //\n"
        "   - Tiếng nói có trung bình lên tới: **mu_sp = 0.20265**, và sigma_sp = 0.23563. //\n"
        "   Sự cách biệt năng lượng này lên tới **hơn 500 lần**! //\n"
        "4. **Bước 4 & 5**: Thiết lập phương trình đẳng xác suất: p(x | Sil) = p(x | Sp). Khai triển logarit, ta được phương trình bậc hai dạng chuẩn: **A*x^2 + B*x + C = 0**. //\n"
        "5. **Bước 6**: Giải phương trình, ta tìm được nghiệm dương duy nhất giữa hai kỳ vọng: **T_Bayes ≈ 0.00288**! // Sau đó áp dụng bộ lọc bắc cầu **200 ms** để loại bỏ các khoảng dừng sinh lý. //",
        "Nghiệm T_Bayes là điểm mà tại đó xác suất phân loại sai thành Speech bằng đúng xác suất phân loại sai thành Silence (Equiprobable Bayesian root), giúp cực tiểu hóa hàm tổn thất 0-1 (Minimum Error Rate Classification)."
    )

    # Slide 13
    add_slide_section(
        13, "ALG 3 · INTERMEDIATE RESULTS", "Parametric Gaussian Fit & Bayes Intersection", "40 – 50 giây",
        "Minh họa hình học giao điểm xác suất và chứng minh tốc độ thực thi tức thời O(1).",
        "Chỉ vào đỉnh Gaussian nhọn bên trái, sau đó trỏ vào tọa độ giao điểm T.",
        "Nhìn vào phân phối hình học, Thầy và các bạn có thể thấy: // Phân phối của khoảng lặng có dáng hình **rất nhọn và hẹp**, ôm sát trục 0. // Trong khi phân phối tiếng nói lại trải rộng về phía bên phải. // Giao điểm xác suất rơi chính xác vào phần đuôi dưới (lower tail) của tiếng nói tại tọa độ **T ≈ 0.00288**. //\n\n"
        "Về mặt hình học quyết định (Decision Geometry): // Vùng diện tích chồng lấn (overlap error area) giữa hai phân phối là **cực kỳ nhỏ**, // nghĩa là xác suất nhận dạng sai theo lý thuyết Bayes đã được **tối thiểu hóa**. //\n\n"
        "Đặc biệt, sau khi đã tìm được nghiệm T ở pha huấn luyện, // ở pha kiểm thử thực tế, thuật toán chỉ cần **so sánh một phép toán O(1) trên từng frame**. // Tốc độ xử lý tức thời và tiêu tốn bộ nhớ gần như bằng 0! //",
        "Độ phức tạp O(1) cho phép triển khai thẳng vào DSP chip hoặc pipeline streaming mà không cần giữ buffer lịch sử như phương pháp Histogram."
    )

    # Slide 14
    add_slide_section(
        14, "ALG 3 · TEST EXECUTION", "Test Boundary Detection & Error (4 Test Files)", "45 – 50 giây",
        "Báo cáo sai số định lượng trên 4 file kiểm thử độc lập và phân tích hiện tượng âm học.",
        "Chỉ vào từng hàng kết quả định lượng, nhấn mạnh con số 7.5 ms.",
        "Kiểm tra mô hình trên **4 file test hoàn toàn mới** (gồm cả môi trường phòng thu Studio và điện thoại thoại Phone): //\n\n"
        "- File **phone_F2**: sai số MAE là **27.5 ms**. //\n"
        "- File **phone_M2**: MAE đạt mức ấn tượng: **7.5 ms**. //\n"
        "- File **studio_F2**: đạt **7.5 ms**. //\n"
        "- File **studio_M2**: cũng đạt **7.5 ms**. //\n"
        "- Trung bình MAE toàn tập test đạt **12.50 ms**! //\n\n"
        "**Nhận xét âm học quan trọng**: // Có tới **3 trên 4 file test** đạt sai số chỉ **7.5 ms**, tức là **dưới 1 bước dịch khung (hop size 10 ms)**! // Ranh giới ở môi trường phòng thu được cắt cực kỳ sắc nét. //\n\n"
        "Riêng file phone_F2, sai số điểm kết thúc trễ **+45 ms**. // Lý do là người nữ phát âm có **tiếng thở hắt ra ở cuối câu (trailing breath)** mang năng lượng lớn hơn T_Bayes, khiến thuật toán kéo dài thêm 4 frame trước khi hạ xuống khoảng lặng. //"
    )

    # Slide 15
    add_slide_section(
        15, "TEST · PHONE F2", "Test Case 1: phone_F2.wav · Comparative VAD", "40 – 45 giây",
        "So sánh chi tiết 3 thuật toán trên trường hợp bị trễ hơi thở cuối câu.",
        "Chỉ vào cột delta End và MAE của 3 thuật toán.",
        "Ở trường hợp kiểm thử số 1: `phone_F2.wav` — kênh thoại băng hẹp 8 kHz, thời lượng 3.02 giây. //\n\n"
        "- **TT1** đạt MAE: **12.5 ms** (đầu lệch -10 ms, cuối lệch +15 ms). //\n"
        "- **TT2** đạt MAE: **17.5 ms**. //\n"
        "- **TT3** đạt MAE: **27.5 ms** (điểm cuối trễ +45 ms như vừa phân tích). //\n\n"
        "Tại sao TT1 lại ít trễ hơn TT3? // Bởi vì ngưỡng tĩnh của TT1 (0.00348) cao hơn ngưỡng Bayes (0.00288). Do đó TT1 đã 'vô tình' cắt đứt tiếng thở sớm hơn, // trong khi TT3 nhạy hơn nên bắt trọn cả phần đuôi hơi thở. //",
        table_data=(
            ["Thuật toán", "ΔStart (ms)", "ΔEnd (ms)", "MAE (ms)"],
            [
                ["TT1 (Fixed STE)", "-10 ms", "+15 ms", "12.5 ms"],
                ["TT2 (Adaptive Hist)", "+20 ms", "-15 ms", "17.5 ms"],
                ["TT3 (Gaussian Bayes)", "-10 ms", "+45 ms", "27.5 ms"]
            ]
        )
    )

    # Slide 16
    add_slide_section(
        16, "TEST · PHONE M2", "Test Case 2: phone_M2.wav · Comparative VAD", "35 – 40 giây",
        "Minh họa độ chính xác cao khi người nói phát âm dứt khoát không có hơi thở thừa.",
        "Chỉ vào con số 7.5 ms của TT1 và TT3.",
        "Sang trường hợp thứ hai: `phone_M2.wav` — giọng nam qua kênh thoại. //\n\n"
        "Ở bản ghi này, năng lượng giọng nói phân tách cực kỳ dứt khoát: //\n"
        "- Cả **TT1** và **TT3** đều đạt kết quả xuất sắc: MAE chỉ **7.5 ms**! // Điểm đầu lệch đúng 1 hop (-10 ms), điểm cuối lệch vỏn vẹn nửa hop (+5 ms). //\n"
        "- **TT2** với cơ chế thích nghi hai đặc trưng đạt MAE là **12.5 ms**. //\n\n"
        "Cả ba thuật toán đều bắt trúng ranh giới, chứng minh rằng khi âm thanh dứt khoát, hiện tượng trôi ranh giới hoàn toàn biến mất! //",
        table_data=(
            ["Thuật toán", "ΔStart (ms)", "ΔEnd (ms)", "MAE (ms)"],
            [
                ["TT1 (Fixed STE)", "-10 ms", "-5 ms", "7.5 ms"],
                ["TT2 (Adaptive Hist)", "+20 ms", "+5 ms", "12.5 ms"],
                ["TT3 (Gaussian Bayes)", "-10 ms", "+5 ms", "7.5 ms"]
            ]
        )
    )

    # Slide 17
    add_slide_section(
        17, "TEST · STUDIO F2", "Test Case 3: studio_F2.wav · Comparative VAD", "35 – 40 giây",
        "Chứng minh ưu thế vượt trội của TT3 trong môi trường phòng thu SNR cao.",
        "Chỉ vào kết quả 7.5 ms của TT3 so với 12.5 ms của TT1 và 17.5 ms của TT2.",
        "Trường hợp thứ ba: `studio_F2.wav` — giọng nữ trong môi trường **phòng thu Studio có SNR rất cao**. //\n\n"
        "Kết quả thể hiện rõ ưu thế lý thuyết: //\n"
        "- **TT3** vượt trội với MAE chỉ **7.5 ms** (điểm kết thúc chỉ lệch -5 ms). //\n"
        "- Trong khi đó, **TT1** đạt **12.5 ms**, // và **TT2** đạt **17.5 ms** do cả hai đều cắt sớm ở cả hai đầu (-15 ms đến -20 ms). //\n\n"
        "Trong phòng thu tĩnh, phân phối của khoảng lặng hầu như không bị nhiễu nền làm méo, giúp ngưỡng Bayes phát huy tối đa độ nhạy lý thuyết! //",
        table_data=(
            ["Thuật toán", "ΔStart (ms)", "ΔEnd (ms)", "MAE (ms)"],
            [
                ["TT1 (Fixed STE)", "-10 ms", "-15 ms", "12.5 ms"],
                ["TT2 (Adaptive Hist)", "-20 ms", "-15 ms", "17.5 ms"],
                ["TT3 (Gaussian Bayes)", "-10 ms", "-5 ms", "7.5 ms"]
            ]
        )
    )

    # Slide 18
    add_slide_section(
        18, "TEST · STUDIO M2", "Test Case 4: studio_M2.wav · Comparative VAD", "35 – 40 giây",
        "Ghi nhận kỷ lục 2.5 ms của TT2 nhờ đệm biên Onset padding.",
        "Chỉ vào con số 2.5 ms và Delta Start = 0 ms của TT2.",
        "Trường hợp kiểm thử cuối: `studio_M2.wav` — giọng nam môi trường phòng thu. //\n\n"
        "- Cả **TT1** và **TT3** đều duy trì sự ổn định tuyệt đối: MAE **7.5 ms**. //\n"
        "- Đặc biệt, **TT2 của bạn Hoàng** đạt mức sai số kỷ lục: **2.5 ms**! // Điểm bắt đầu trùng khớp tuyệt đối **±0 ms** với Ground-Truth, điểm kết thúc chỉ lệch đúng -5 ms! //\n\n"
        "Nhờ cơ chế đệm biên **+10 ms padding**, TT2 đã khôi phục hoàn hảo phụ âm đầu mà không làm biến dạng ranh giới tổng thể. //",
        table_data=(
            ["Thuật toán", "ΔStart (ms)", "ΔEnd (ms)", "MAE (ms)"],
            [
                ["TT1 (Fixed STE)", "+10 ms", "+5 ms", "7.5 ms"],
                ["TT2 (Adaptive Hist)", "±0 ms", "-5 ms", "2.5 ms"],
                ["TT3 (Gaussian Bayes)", "+10 ms", "+5 ms", "7.5 ms"]
            ]
        )
    )

    # Slide 19
    add_slide_section(
        19, "BENCHMARK · 4 TEST FILES", "Quantitative Performance Comparison Across Algorithms", "75 – 90 giây",
        "Trình bày bảng benchmark tổng hợp định lượng toàn diện và rút ra kết luận cốt lõi.",
        "Đứng thẳng, tay hướng trọn vẹn vào Bảng Benchmark, đọc từng chỉ số chính xác.",
        "Kính thưa Thầy, đây là **bức tranh toàn cảnh định lượng** của cả ba thuật toán trên toàn bộ 4 file kiểm thử: //\n\n"
        "1. **TT1-2 (Hodgkinson cải tiến)**: Đạt sai số biên thấp nhất: **MAE = 6.25 ms**, RMSE = 7.80 ms, điểm F1 = **0.996**. //\n"
        "2. **TT2-1 (Histogram có đệm biên +10ms)**: Đạt MAE **12.50 ms**, cải thiện vượt bậc **33.3%** so với bản không đệm (18.75 ms). //\n"
        "3. **TT3-1 (Gaussian 1D của em)**: Đạt MAE **12.50 ms**, F1 = **0.994**, độ nhạy âm vô thanh UV đạt **0.971**! //\n\n"
        "Một phát hiện thực nghiệm cực kỳ đắt giá: // Khi nhóm thử nghiệm phiên bản **4D Multivariate Gaussian** kết hợp phổ, // sai số lại **tăng lên 15.00 ms**, kém hơn bản 1D! //\n\n"
        "**Kết luận cốt lõi**: TT1-2 đạt sai số biên thấp nhất nhờ tối ưu trực tiếp thời lượng. Nhưng TT3 chứng minh sức mạnh toán học khi **3 trên 4 file test đạt MAE 7.5 ms** hoàn toàn bằng suy diễn giải tích đóng! //",
        table_data=(
            ["Phương pháp", "MAE (ms)", "RMSE (ms)", "F1-Score", "Độ nhạy UV"],
            [
                ["TT1-1 (Hodgkinson gốc)", "10.00", "10.33", "0.993", "0.971"],
                ["TT1-2 (DSP Classifier)", "6.25", "7.80", "0.996", "0.968"],
                ["TT2-1 (Dual STE-SC + Pad)", "12.50", "13.37", "0.985", "0.947"],
                ["TT2-2 (Baseline No-Pad)", "18.75", "19.34", "0.977", "0.891"],
                ["TT3-1 (1D Gaussian Bayes)", "12.50", "14.08", "0.994", "0.971"],
                ["TT3-2 (4D Multivariate)", "15.00", "16.34", "0.978", "0.903"]
            ]
        )
    )

    # Slide 20
    add_slide_section(
        20, "ALG 1 · CRITIQUE", "Strengths, Weaknesses, and Best Use Case", "45 giây",
        "Phê bình khách quan ưu điểm, nhược điểm và môi trường áp dụng của TT1.",
        "Chia tay làm 3 hướng tương ứng 3 cột trên slide.",
        "Phê bình chuyên sâu **Thuật toán 1 (Ngưỡng tĩnh toàn cục)**: //\n\n"
        "- **Ưu điểm**: Kiểm tra khung với độ phức tạp O(1), không tốn RAM đệm, tốc độ nhanh nhất và độ chính xác ranh giới cao nhất khi nhiễu dừng. //\n"
        "- **Nhược điểm chí mạng**: Ngưỡng bị đóng cứng (rigid static). Khi môi trường có nhiễu nền biến động hoặc SNR tụt dốc, thuật toán mất hoàn toàn khả năng thích nghi. //\n"
        "- **Ứng dụng lý tưởng**: Phù hợp cho **vi điều khiển nhúng công suất thấp (MCU)**, chip IoT biên, hoặc hệ thống streaming thời gian thực trong phòng thu tĩnh. //\n\n"
        "**Bài học rút ra**: *Đơn giản chính là đỉnh cao khi môi trường nhiễu có tính dừng!* //"
    )

    # Slide 21
    add_slide_section(
        21, "ALG 2 · CRITIQUE", "Dual-Feature Strengths, Weaknesses, and Best Use Case", "45 giây",
        "Phê bình ưu nhược điểm của TT2: thế mạnh không giám sát và nhược điểm trên câu ngắn.",
        "Hướng tay sang phân tích lược đồ Histogram 2 đặc trưng.",
        "Đối với **Thuật toán 2 (Lược đồ Histogram thích nghi)**: //\n\n"
        "- **Ưu điểm vượt trội**: Hoàn toàn **không giám sát (Unsupervised)**! Tự thích nghi trên từng câu nói nhờ kết hợp cả năng lượng thời gian (STE) và trọng tâm phổ tần số (Spectral Centroid). Không cần dữ liệu gán nhãn trước. //\n"
        "- **Nhược điểm**: Bắt buộc phân phối phải có dạng **hai đỉnh (Bimodal)**. Với câu nói quá ngắn dưới 1 giây, hoặc SNR quá thấp làm bẹp đáy thung lũng giữa 2 đỉnh, thuật toán sẽ chọn sai ngưỡng. Phải giữ toàn bộ câu vào RAM (O(N)). //\n"
        "- **Ứng dụng thực tế**: Cực kỳ phù hợp cho **hệ thống xử lý âm thanh tự động ngoại tuyến (Offline)**, cắt gọt khoảng lặng cho Podcast, lập chỉ mục kho âm thanh quy mô lớn. //\n\n"
        "**Bài học**: *Lược đồ thời gian - tần số mang lại ngưỡng tự thích nghi mà không cần nhãn!* //"
    )

    # Slide 22
    add_slide_section(
        22, "ALG 3 · CRITIQUE", "Model Complexity: 1D Gaussian vs 4D Covariance Matrix", "50 giây",
        "Giải thích nguyên lý Dao cạo Occam (Occam's Razor) – vì sao mô hình 1D đánh bại 4D.",
        "Nhìn thẳng Hội đồng, nhấn mạnh giọng triết lý khoa học.",
        "Đối với **Thuật toán 3**, nhóm đặt ra câu hỏi học thuật: // **Liệu tăng độ phức tạp mô hình lên đa biến 4D Gaussian có giúp VAD tốt hơn không?** //\n\n"
        "Kết quả thực nghiệm chứng minh một bài học kinh điển: //\n"
        "- **Mô hình 1D Gaussian** chỉ có 4 tham số, nghiệm giải tích đóng ổn định tuyệt đối, đạt MAE **12.50 ms**. //\n"
        "- **Mô hình 4D Multivariate** phải ước lượng ma trận hiệp phương sai đầy đủ. Khi tập huấn luyện chỉ có 4 file (cỡ mẫu nhỏ), mô hình bị **quá khớp (Overfitting)** và gặp rủi ro số học khi nghịch đảo ma trận, khiến MAE tăng lên **15.00 ms**! //\n\n"
        "**Triết lý Dao cạo Occam (Occam’s Razor)**: // Trong điều kiện dữ liệu mẫu nhỏ, **sự tối giản tham số (Parsimony) luôn đánh bại sự phức tạp hóa mô hình**! //"
    )

    # Slide 23
    add_slide_section(
        23, "PREPROCESSING · STANDARDS", "Standard Parameters & 200 ms Silence Bridging", "45 giây",
        "Khẳng định tính chuẩn mực kỹ thuật và quy tắc bắc cầu 200ms bảo vệ ngữ nghĩa.",
        "Chỉ vào các thông số chuẩn hóa DSP trên màn hình.",
        "Để kết quả so sánh giữa 3 thành viên đạt độ khách quan khoa học, // cả nhóm đã thống nhất một **chuẩn tiền xử lý DSP đồng nhất**: //\n\n"
        "1. **Tần số lấy mẫu Fs = 16 kHz**: Chuẩn hóa toàn bộ âm thanh về mono 16 kHz. //\n"
        "2. **Khung 25 ms** (400 mẫu): Đảm bảo tính chất chuẩn dừng (quasi-stationary) của bộ máy phát âm người. //\n"
        "3. **Bước nhảy 10 ms** (160 mẫu): Tạo độ chồng lấn 60%, thiết lập lưới thời gian chính xác 10 ms. //\n"
        "4. **Quy tắc bắc cầu bất biến (Bridging Invariant)**: //\n"
        "   - Khoảng lặng ngắn hơn **200 ms** sẽ bị **bắc cầu nối liền (Bridge)**, // vì về mặt ngữ âm, đó chỉ là khoảng dừng lấy hơi hoặc đóng thanh quản tạm thời giữa các âm tiết. //\n"
        "   - Đoạn tiếng nói ngắn hơn **50 ms** bị **loại bỏ hoàn toàn**, // để triệt tiêu tiếng lách cách (click noise) do micro va chạm! //"
    )

    # Slide 24
    add_slide_section(
        24, "LIVE DEMO · SUBMISSION", "Four-Quadrant Display on Teacher Laptop", "40 giây",
        "Trình bày quy cách nộp bài và sẵn sàng demo trực tiếp 4 góc phần tư trên laptop Thầy.",
        "Chỉ vào 4 khối 01 - 02 - 03 - 04.",
        "Về quy cách nộp bài và tổ chức thực nghiệm: //\n\n"
        "- **Khối 01**: Dữ liệu đầu vào gồm 4 file test `.wav` và file nhãn chuẩn `.lab`. Không upload âm thanh lên mạng hay lưu trữ dư thừa. //\n"
        "- **Khối 02**: Môi trường ảo được quản lý chuẩn hóa qua **công cụ uv**, Python 3.12, dùng thư viện toán học thuần numpy và scipy. //\n"
        "- **Khối 03**: Ba file Jupyter Notebook độc lập đều đã được chạy sẵn kết quả (Pre-run outputs), đồ thị sẵn sàng, **độ trễ demo bằng 0**! //\n"
        "- **Khối 04**: Giao diện được thiết kế theo **bố cục 4 góc phần tư (Four-Quadrant Display)** trên màn hình laptop của Thầy, giúp đối chiếu đồng thời cả dạng sóng, đường STE và nhãn phân đoạn cùng lúc! //"
    )

    # Slide 25
    add_slide_section(
        25, "DEFENSE · Q&A", "Rebuttal for Critical Examination Questions", "80 – 90 giây",
        "Chủ động phản biện 4 câu hỏi hóc búa nhất của Hội đồng đánh giá.",
        "Đứng vững chãi, hướng thẳng về Thầy, nói dõng dạc, rành mạch từng câu hỏi.",
        "Trước khi Thầy đặt câu hỏi, nhóm xin chủ động giải trình **4 vấn đề then chốt** mà Hội đồng thường quan tâm nhất: //\n\n"
        "**Câu hỏi 1: Tại sao nhóm không dùng tần số cơ bản F0 (Pitch) để phân đoạn tiếng nói?** //\n"
        "👉 **Trả lời**: Bởi vì các **phụ âm vô thanh (Unvoiced consonants)** như /s/, /t/, /f/ hoàn toàn **không có tần số cơ bản F0**! // Nếu dùng F0, chúng ta sẽ cắt cụt toàn bộ các âm đầu và âm cuối của từ, phá hủy ngữ nghĩa câu nói! //\n\n"
        "**Câu hỏi 2: Tại sao ở file phone_M2, có thuật toán đạt sai số gần như 0.0 ms? Có phải do Overfitting?** //\n"
        "👉 **Trả lời**: Hoàn toàn **không phải Overfitting**! // Nhãn Ground-truth [0.53s, 2.52s] trùng khớp chính xác vào khung 53 và khung 252 trên lưới bước nhảy 10 ms. Đây là hiện tượng **khớp lưới bước nhảy (Grid snapping)**, chứng minh độ phân giải lưới 10 ms hoạt động cực chuẩn xác! //\n\n"
        "**Câu hỏi 3: Mức nhiễu nền SNR ảnh hưởng như thế nào đến độ dịch ranh giới?** //\n"
        "👉 **Trả lời**: Ở Studio (SNR cao), sai số chỉ 5–10 ms. Nhưng ở Phone (băng hẹp 8kHz, SNR thấp), tiếng thở hắt đuôi câu tạo ra độ trễ từ 20–45 ms. //\n\n"
        "**Câu hỏi 4: Tại sao 1D Gaussian lại đánh bại đa biến 4D trong TT3?** //\n"
        "👉 **Trả lời**: Tập huấn luyện chỉ có 4 file. Ước lượng ma trận hiệp phương sai 4D gây Overfitting nặng. Mô hình 1D ít tham số hơn nên có tính **tổng quát hóa vượt trội** trên tập kiểm thử! //"
    )

    # Slide 26
    add_slide_section(
        26, "DIGITAL SIGNAL PROCESSING · MIDTERM PROJECT 2026", "Thank You & Q&A Session", "25 – 30 giây",
        "Kết thúc bài báo cáo lịch sự, trang trọng và mở phiên Live Demo / Vấn đáp.",
        "Cúi đầu chào nhẹ, hai tay mở hướng về Thầy và lớp học.",
        "Kính thưa Thầy và toàn thể các bạn! //\n\n"
        "Trên đây là toàn bộ báo cáo kết quả nghiên cứu đề tài **Phân đoạn tín hiệu tiếng nói – khoảng lặng (Voice Activity Detection)** của Nhóm 08. //\n"
        "Mã nguồn, sổ tay tính toán Notebook và các biểu đồ định lượng đều đã sẵn sàng trong thư mục nộp bài. //\n"
        "Nhóm chúng em đã sẵn sàng cho phần **Live Demo trực tiếp trên bất kỳ file âm thanh nào** mà Thầy yêu cầu, // và rất mong nhận được những nhận xét, góp ý quý báu từ Thầy! //\n\n"
        "**Em xin trân trọng cảm ơn Thầy và các bạn đã chú ý lắng nghe!** //"
    )

    doc.save(filename)
    print(f"Document saved successfully: {filename}")

if __name__ == "__main__":
    import os
    out_dir = "08-Phân đoạn tín hiệu tiếng nói - khoảng lặng"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Script_Thuyet_Trinh_Slide_11_26.docx")
    build_docx(out_path)
