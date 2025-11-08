# QITSA: Phân tích Cảm xúc Văn bản theo Cảm hứng Lượng tử - Hướng dẫn Huấn luyện Đầy đủ cho Nhà phát triển

## Tổng quan Dự án

**QITSA** (Kiến trúc Học sâu Có thể Diễn giải theo Cảm hứng Lượng tử cho Phân tích Cảm xúc Văn bản) là một framework học sâu mang tính cách mạng, kết hợp các nguyên lý cơ học lượng tử với xử lý ngôn ngữ tự nhiên để đạt được hiệu suất phân tích cảm xúc tiên tiến nhất. Phương pháp sáng tạo này không chỉ mang lại độ chính xác vượt trội trên các bộ dữ liệu chuẩn mà còn cung cấp khả năng diễn giải chưa từng có trong việc hiểu cách thức đưa ra dự đoán cảm xúc.

### Điểm đặc biệt của QITSA

**Đổi mới theo Cảm hứng Lượng tử**: Khác với các mạng nơ-ron truyền thống xử lý văn bản như các vector đơn giản, QITSA biểu diễn từ như các trạng thái lượng tử với cả thành phần thực và ảo, cho phép biểu diễn phong phú hơn về thông tin ngữ nghĩa và cảm xúc.

**Khả năng Diễn giải Nâng cao**: Kiến trúc lấy cảm hứng từ lượng tử cho phép trực quan hóa cách các từ khác nhau đóng góp vào dự đoán cảm xúc, giúp có thể hiểu được "lý do" đằng sau mỗi dự đoán.

**Xuất sắc trên Nhiều Bộ dữ liệu**: Đạt hiệu suất vượt trội trên năm bộ dữ liệu chuẩn phân tích cảm xúc, chứng minh tính bền vững và khả năng tổng quát hóa.

**Kiến trúc Mới lạ**: Kết hợp các thành phần học sâu truyền thống (CNN, embeddings) với các phép biến đổi theo cảm hứng lượng tử cho một phương pháp lai độc đáo.

### Nền tảng Nghiên cứu
- **Bài báo**: [ArXiv 2408.07891](http://arxiv.org/abs/2408.07891)  
- **Framework**: PyTorch 2.0+
- **Ngôn ngữ lập trình**: Python 3.8+
- **Giấy phép**: MIT
- **Trọng tâm Nghiên cứu**: NLP theo cảm hứng lượng tử, AI có thể diễn giải, Phân tích cảm xúc

### Các Đổi mới Kỹ thuật Chính
1. **Biểu diễn Số phức**: Từ được mã hóa như trạng thái lượng tử với thành phần thực và ảo
2. **Chồng chập Lượng tử**: Nhiều trạng thái cảm xúc có thể cùng tồn tại trước khi đo
3. **Giao thoa Lượng tử**: Các mẫu giao thoa tạo dựng/triệt tiêu nâng cao độ rõ tín hiệu cảm xúc
4. **Đo lường Lượng tử Có thể Diễn giải**: Trực quan hóa sự sụp đổ trạng thái lượng tử trong quá trình dự đoán

## Phân tích Cấu trúc Dự án Đầy đủ

```
QITSA/
├── Các File Huấn luyện Cốt lõi
│   ├── aMain.py.ipynb          # ĐIỂM KHỞI ĐẦU CHÍNH - Notebook huấn luyện chính
│   ├── train.py                         # Script huấn luyện độc lập
│   └── QITSA_Complete_Training_Guide.md # Tài liệu gốc tiếng Anh
│
├── Trực quan hóa & Kiểm tra
│   ├── test_vis.py                      # Công cụ trực quan hóa chung
│   └── test_vis_english.py              # Công cụ trực quan hóa tiếng Anh
│
├── Bộ sưu tập Dữ liệu (5 Bộ dữ liệu Chuẩn)
│   ├── CR/ (Customer Reviews)           # Phân tích cảm xúc đánh giá sản phẩm
│   │   ├── train.tsv (3,775 mẫu)       # Dữ liệu huấn luyện
│   │   ├── dev.tsv (1,259 mẫu)         # Dữ liệu phát triển/xác thực
│   │   └── test.tsv (1,259 mẫu)        # Đánh giá kiểm tra cuối cùng
│   ├── MPQA/ (Multi-Perspective QA)     # Phát hiện cực tính ý kiến
│   │   ├── train.tsv (7,293 mẫu)
│   │   ├── dev.tsv (1,823 mẫu)
│   │   └── test.tsv (1,823 mẫu)
│   ├── MR/ (Movie Reviews)              # Phân loại cảm xúc phim
│   │   ├── train.tsv (8,530 mẫu)
│   │   ├── dev.tsv (1,066 mẫu)
│   │   └── test.tsv (1,066 mẫu)
│   ├── SST/ (Stanford Sentiment Treebank) # Phân tích cảm xúc chi tiết
│   │   ├── train.tsv (67,349 mẫu)      # Bộ dữ liệu lớn nhất
│   │   ├── dev.tsv (872 mẫu)
│   │   └── test.tsv (1,821 mẫu)
│   └── SUBJ/ (Subjectivity Dataset)     # Phân loại Chủ quan vs Khách quan
│       ├── train.tsv (8,000 mẫu)
│       ├── dev.tsv (1,000 mẫu)
│       └── test.tsv (1,000 mẫu)
│
├── Kết quả & Trực quan hóa
│   └── figs/                            # Các biểu đồ huấn luyện được tạo ra
│
└── Lưu trữ Thực nghiệm
    └── Main_results/           # Các biến thể nghiên cứu
        ├── 1/ (Thực nghiệm Kiến trúc)
        ├── 2/ (Thực nghiệm Biểu diễn Đầu vào)
        └── 3/ (Thực nghiệm Nâng cao)
```

## Phân tích Toàn diện về Bộ dữ liệu

### Đặc điểm và Ứng dụng của Bộ dữ liệu

#### Bảng Tổng quan Bộ dữ liệu

| Bộ dữ liệu | Tên đầy đủ | Loại tác vụ | Kích thước | Lĩnh vực | Nhãn | Ứng dụng |
|---------|-----------|-----------|------|--------|--------|-------------|
| **MR** | Movie Reviews | Phân loại Cảm xúc Nhị phân | ~10K | Giải trí | 0=Tiêu cực, 1=Tích cực | Phân tích phê bình phim |
| **CR** | Customer Reviews | Phân loại Cảm xúc Nhị phân | ~6K | Thương mại điện tử | 0=Tiêu cực, 1=Tích cực | Phân tích phản hồi sản phẩm |
| **SST** | Stanford Sentiment Treebank | Phân loại Cảm xúc Chi tiết | ~70K | Phim | 0=Tiêu cực, 1=Tích cực | Điểm chuẩn học thuật |
| **SUBJ** | Subjectivity Dataset | Phát hiện Tính chủ quan | ~10K | Đa dạng | 0=Khách quan, 1=Chủ quan | Phân loại ý kiến vs sự kiện |
| **MPQA** | Multi-Perspective QA | Cực tính Ý kiến | ~11K | Tin tức/Chính trị | 0=Tiêu cực, 1=Tích cực | Khai thác ý kiến đa quan điểm |

### Mô tả Chi tiết Bộ dữ liệu

#### 1. **MR (Movie Reviews)** - Tập trung vào Ngành Giải trí
- **Mục đích**: Phân loại đánh giá phim là tích cực hay tiêu cực
- **Đặc điểm Văn bản**: Đánh giá ngắn đến trung bình (10-30 từ)
- **Phân bố Nhãn**: Cân bằng cảm xúc tích cực/tiêu cực
- **Ứng dụng Thực tế**: Hệ thống đề xuất phim, dự đoán doanh thu phòng vé
- **Độ phức tạp Mẫu**: Trung bình - chứa ý kiến chủ quan và tham chiếu văn hóa

#### 2. **CR (Customer Reviews)** - Phân tích Thương mại điện tử & Sản phẩm  
- **Mục đích**: Phân tích mức độ hài lòng của khách hàng từ đánh giá sản phẩm
- **Đặc điểm Văn bản**: Đánh giá độ dài trung bình (15-40 từ)
- **Phân bố Nhãn**: Cân bằng đánh giá tích cực/tiêu cực
- **Ứng dụng Thực tế**: Nền tảng thương mại điện tử, phản hồi cải tiến sản phẩm
- **Độ phức tạp Mẫu**: Cao - chứa thuật ngữ kỹ thuật và biệt ngữ riêng về sản phẩm

#### 3. **SST (Stanford Sentiment Treebank)** - Điểm chuẩn Học thuật
- **Mục đích**: Phân tích cảm xúc chi tiết với chú thích cấp cụm từ
- **Đặc điểm Văn bản**: Cấp cụm từ đến cấp câu (5-50 từ)
- **Phân bố Nhãn**: Ban đầu 5 lớp, được chuyển đổi thành nhị phân
- **Ứng dụng Thực tế**: Nghiên cứu học thuật, điểm chuẩn phân tích cảm xúc
- **Độ phức tạp Mẫu**: Rất cao - bao gồm các cấu trúc ngôn ngữ phức tạp

#### 4. **SUBJ (Subjectivity Dataset)** - Phân tích Tính khách quan
- **Mục đích**: Phân biệt giữa ý kiến chủ quan và tuyên bố khách quan
- **Đặc điểm Văn bản**: Các câu có độ dài biến đổi (10-60 từ)
- **Phân bố Nhãn**: 50% chủ quan / 50% khách quan
- **Ứng dụng Thực tế**: Phân tích tin tức, hệ thống kiểm tra sự thật
- **Độ phức tạp Mẫu**: Cao - yêu cầu hiểu các dấu hiệu ngôn ngữ về tính chủ quan

#### 5. **MPQA (Multi-Perspective Question Answering)** - Khai thác Ý kiến
- **Mục đích**: Phát hiện cực tính ý kiến đa quan điểm
- **Đặc điểm Văn bản**: Cụm từ và câu ngắn (3-25 từ)
- **Phân bố Nhãn**: Cực tính ý kiến tích cực/tiêu cực
- **Ứng dụng Thực tế**: Phân tích chính trị, giám sát mạng xã hội
- **Độ phức tạp Mẫu**: Rất cao - chứa cảm xúc ngầm và ý nghĩa phụ thuộc ngữ cảnh

### Đặc tả Định dạng Dữ liệu

Tất cả các bộ dữ liệu tuân theo định dạng **TSV (Tab-Separated Values)** nhất quán:
- **Cột 1**: Nội dung văn bản đã tiền xử lý (được phân đoạn, chuyển thành chữ thường)
- **Cột 2**: Nhãn nhị phân (0 = tiêu cực/khách quan, 1 = tích cực/chủ quan)
- **Mã hóa**: UTF-8
- **Ký tự phân cách**: Ký tự tab (`\t`)
- **Tiêu đề**: Không có dòng tiêu đề (dữ liệu bắt đầu từ dòng 1)

### Ví dụ Dữ liệu Mẫu Chi tiết

**Đánh giá Khách hàng (CR)**:
```tsv
navigation is so smooth and finding files is a cinch .	1
the volume level of the phone is not all that good .	0
sound quality is great for the price range .	1
battery life could be much better than expected .	0
```

**Đánh giá Phim (MR)**:
```tsv
there 's lots of cool stuff packed into espn 's ultimate x .	1
madonna still ca n't act a lick .	0
enormously enjoyable , high-adrenaline documentary .	1
despite some comic sparks , welcome to collinwood never catches fire .	0
```

**Tính chủ quan (SUBJ)**:
```tsv
carefully crafted , notably in its deft dramatic structuring .	1  (ý kiến chủ quan)
five hundred years before columbus , a young norwegian viking discovers america .	0  (sự kiện khách quan)
it has plenty of laughs but lacks emotional depth .	1  (đánh giá chủ quan)
the story follows a detective investigating a murder case .	0  (mô tả khách quan)
```

### Thống kê và Thông tin Tiền xử lý

#### Quy trình Tiền xử lý Văn bản:
1. **Phân đoạn từ (Tokenization)**: Tách cấp từ với xử lý dấu câu
2. **Chuyển sang chữ thường**: Tất cả văn bản được chuyển thành chữ thường
3. **Ký tự Đặc biệt**: Được giữ lại cho các chỉ báo cảm xúc (!, ?, v.v.)
4. **Chuẩn hóa Độ dài**: Các câu được cắt bớt/đệm khi cần thiết
5. **Mã hóa**: Mã hóa UTF-8 cho hỗ trợ ký tự quốc tế

#### Phân tích Thống kê:
- **Độ dài Câu Trung bình**: 15-25 từ trên các bộ dữ liệu
- **Kích thước Từ vựng**: 10K-50K từ độc nhất mỗi bộ dữ liệu
- **Cân bằng Nhãn**: Tất cả các bộ dữ liệu duy trì phân bố lớp ~50/50
- **Ngôn ngữ**: Tiếng Anh (với một số bộ dữ liệu chứa thành ngữ)

## Thiết lập Môi trường Phát triển Đầy đủ

### Yêu cầu Hệ thống
```yaml
Hệ điều hành: Windows 10/11, Linux Ubuntu 18+, macOS 10.15+
Phiên bản Python: 3.8 - 3.11 (khuyến nghị 3.9)
RAM: Tối thiểu 8GB (khuyến nghị 16GB+ cho bộ dữ liệu lớn)
GPU: NVIDIA GPU có hỗ trợ CUDA (tùy chọn nhưng rất khuyến nghị)
Dung lượng: 5GB dung lượng trống cho bộ dữ liệu và mô hình
Internet: Yêu cầu cho việc tải gói ban đầu
```

### Hướng dẫn Cài đặt Từng bước

#### Bước 1: Thiết lập Môi trường Python
```bash
# Tạo môi trường ảo độc lập (KHUYẾN NGHỊ MẠNH MẼ)
python -m venv qitsa_env

# Kích hoạt môi trường
# Windows (PowerShell):
qitsa_env\Scripts\Activate.ps1
# Windows (Command Prompt):
qitsa_env\Scripts\activate.bat
# Linux/macOS:
source qitsa_env/bin/activate

# Nâng cấp pip lên phiên bản mới nhất
python -m pip install --upgrade pip
```

#### Bước 2: Cài đặt Các Thư viện Cốt lõi
```bash
# Cài đặt PyTorch với hỗ trợ CUDA (khuyến nghị)
# Cho CUDA 11.8 (kiểm tra phiên bản CUDA: nvcc --version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Để cài đặt chỉ CPU (nếu không có GPU CUDA)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các gói tính toán khoa học cần thiết
pip install numpy pandas matplotlib seaborn plotly
pip install scikit-learn scipy nltk tqdm

# Cài đặt Jupyter để sử dụng notebook
pip install jupyter notebook ipykernel jupyterlab

# Cài đặt các tiện ích bổ sung
pip install wordcloud pillow requests
```

#### Bước 3: Tải Dữ liệu NLTK
```python
# Chạy script Python này để tải các bộ dữ liệu NLTK cần thiết
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Tải dữ liệu NLTK cần thiết
nltk.download('wordnet')        # Cho ý nghĩa từ và mối quan hệ
nltk.download('sentiwordnet')   # Cho điểm cực tính cảm xúc
nltk.download('punkt')          # Cho phân đoạn từ
nltk.download('stopwords')      # Cho lọc từ phổ biến
nltk.download('averaged_perceptron_tagger')  # Cho gắn thẻ từ loại

print("Đã tải thành công tất cả dữ liệu NLTK!")
```

#### Bước 4: Xác minh Thiết lập GPU (Tùy chọn nhưng Khuyến nghị)
```python
import torch

print("Thông tin Hệ thống:")
print(f"Phiên bản Python: {torch.__version__}")
print(f"Phiên bản PyTorch: {torch.__version__}")
print(f"CUDA có sẵn: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Phiên bản CUDA: {torch.version.cuda}")
    print(f"Số lượng GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Bộ nhớ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA không có sẵn - sẽ sử dụng CPU (huấn luyện sẽ chậm hơn)")
```

### Yêu cầu Dữ liệu Bên ngoài

#### Các Bản tải về Bắt buộc:
1. **Bộ dữ liệu WordNet**: [Liên kết Tải về](https://drive.google.com/file/d/1zDPySBmsagwY1jIqzYejOOci-oFe51oe/view?usp=sharing)
   - Chứa các mối quan hệ ngữ nghĩa giữa các từ
   - Cần thiết cho việc hiểu cảm xúc nâng cao
   - Giải nén vào thư mục gốc của dự án

2. **DataFormBERT**: [Liên kết Tải về](https://drive.google.com/file/d/1iwY1CcRdRH3WnANpascAj6q9MREkawlU/view?usp=sharing)
   - Các biểu diễn dữ liệu được tiền xử lý tương thích BERT
   - Được sử dụng cho các thí nghiệm biểu diễn đầu vào nâng cao
   - Giải nén vào thư mục gốc của dự án

## Bắt đầu: Lần Chạy Huấn luyện Đầu tiên

### Khởi động Nhanh (5 Phút đến Kết quả Đầu tiên)

#### Tùy chọn 1: Sử dụng Jupyter Notebook (Khuyến nghị)

**Bước 1: Khởi chạy Jupyter**
```bash
# Đảm bảo bạn đang ở trong thư mục QITSA
cd d:\ThacSy\PROJECTS\QITSA

# Khởi động Jupyter Notebook
jupyter notebook
```

**Bước 2: Mở Notebook Huấn luyện Chính**
- Điều hướng đến và mở: `aMain.py.ipynb`
- Đây là điểm khởi đầu chính của bạn để huấn luyện

**Bước 3: Cấu hình Thí nghiệm Đầu tiên**

Trong vài ô đầu tiên của notebook, bạn sẽ tìm thấy các biến cấu hình chính:

```python
# ===============================
# CÀI ĐẶT CẤU HÌNH CỐT LÕI
# ===============================

# Lựa chọn Bộ dữ liệu (THAY ĐỔI để thử các bộ dữ liệu khác nhau)
DATA_SET = 'CR'        # Bắt đầu với 'CR' (Đánh giá Khách hàng) - bộ dữ liệu nhỏ nhất
                       # Các tùy chọn: 'MR', 'CR', 'SST', 'SUBJ', 'MPQA'

# Tham số Huấn luyện
BATCH_SIZE = 16        # Giảm xuống 8 nếu gặp vấn đề bộ nhớ
N_EPOCHS = 26          # Số epoch huấn luyện
learning_rate = 0.001  # Tốc độ học của bộ tối ưu hóa Adam

# Tham số Kiến trúc Mô hình
word_dim = 100         # Chiều embedding từ
EMBEDDING_DIM = 100    # Phải khớp với word_dim
HIDDEN_DIM = 3         # Chiều ẩn lượng tử (nhỏ hơn = nhanh hơn)
OUTPUT_DIM = 1         # Đầu ra phân loại nhị phân
num_layers = 2         # Số lớp biến đổi lượng tử

# Cài đặt Hệ thống
CUDA_NUMBER = 0        # Số thiết bị GPU (0 cho GPU đầu tiên, -1 cho CPU)
PAD_IDX = 1           # Chỉ số token đệm
```

**Bước 4: Chạy Tất cả Các Ô**
- Thực thi từng ô theo thứ tự bằng cách nhấn `Shift + Enter`
- Chú ý đến bất kỳ thông báo lỗi nào

#### Tùy chọn 2: Sử dụng Script Huấn luyện

```bash
# Chạy script huấn luyện trực tiếp
python train.py

# Chuyển hướng đầu ra sang file log
python train.py > training_log.txt 2>&1
```

### Hiểu về Lựa chọn Bộ dữ liệu

```python
# Từ điển Ánh xạ Bộ dữ liệu
dataset_dict = {
    'MR': 0,      # Đánh giá Phim
    'CR': 1,      # Đánh giá Khách hàng
    'SST': 2,     # Stanford Sentiment Treebank
    'SUBJ': 3,    # Bộ dữ liệu Tính chủ quan
    'MPQA': 4     # Hỏi đáp Đa quan điểm
}

# Đường dẫn tự động được cấu hình
DATA_SET = 'CR'
train_path = f"./DataSet/{DATA_SET}/train.tsv"
test_path = f"./DataSet/{DATA_SET}/test.tsv"
dev_path = f"./DataSet/{DATA_SET}/dev.tsv"
word_vector_path = f"./sub_word_vector/{DATA_SET}_word_vector.txt"
```

### Khuyến nghị cho Người mới bắt đầu

**Thứ tự Thử nghiệm Khuyến nghị:**

1. **Bắt đầu với 'CR'**: Nhỏ nhất, nhanh nhất (~5-10 phút)
2. **Sau đó 'MR'**: Trung bình, ngôn ngữ phức tạp hơn
3. **Nâng cao 'SST'**: Lớn nhất, thử thách nhất (~20-60 phút)
4. **Chuyên biệt 'SUBJ'**: Tác vụ khác (khách quan vs chủ quan)
5. **Chuyên gia 'MPQA'**: Phức tạp, đa quan điểm

### Giải thích Tham số Cấu hình

```python
# Tham số Điều khiển Huấn luyện
BATCH_SIZE = 16        # Số câu xử lý đồng thời
                       # Lớn hơn = nhanh hơn nhưng dùng nhiều bộ nhớ hơn
                       # Giảm xuống 8 hoặc 4 nếu lỗi "CUDA out of memory"

N_EPOCHS = 26         # Số lần duyệt qua toàn bộ dữ liệu
                      # Nhiều epoch = huấn luyện lâu hơn, có thể tốt hơn
                      # Bắt đầu với 26, có thể giảm xuống 10 để kiểm tra nhanh

learning_rate = 0.001 # Tốc độ học của mô hình
                      # Quá cao = không ổn định, quá thấp = chậm
                      # 0.001 là mặc định tốt cho Adam

# Tham số Kiến trúc
EMBEDDING_DIM = 100   # Số chiều của vector từ
                      # Cao hơn = biểu đạt nhiều hơn nhưng chậm hơn
                      # Phải khớp với embedding tiền huấn luyện

HIDDEN_DIM = 3        # Chiều trạng thái lượng tử (độc đáo của QITSA)
                      # Kiểm soát độ phức tạp lượng tử
                      # Bắt đầu với 3, thử 5-10 cho mô hình lớn

num_layers = 2        # Số lớp biến đổi lượng tử
                      # Nhiều lớp = phức tạp hơn nhưng chậm hơn
                      # 2-3 lớp thường tối ưu
```

### Điều gì Mong đợi Trong Huấn luyện

#### Đầu ra Console:

```bash
Đang xử lý bộ dữ liệu: ./DataSet/CR/train.tsv
Giai đoạn 1: Phân tích các câu...
Đã tải 3775 câu từ ./DataSet/CR/train.tsv

Giai đoạn 2: Chuyển đổi từ thành chỉ số...
Tìm thấy 45 từ không xác định (sử dụng token UNK)

Giai đoạn 3: Tính toán điểm cảm xúc...
100%|████████████| 3775/3775 [00:30<00:00, 125.2it/s]

Giai đoạn 4: Đệm các câu...
Giai đoạn 5: Tạo trình lặp batch...
Đã tạo 236 batch có kích thước 16

Mô hình được khởi tạo với 892,547 tham số có thể huấn luyện
Đang sử dụng thiết bị: cuda:0

Bắt đầu huấn luyện cho 26 epoch...

Epoch 1/26 [Train]: 100%|████████| 236/236 [00:45<00:00, 5.2batch/s, Loss=0.6234, Acc=0.6543]
Epoch 1/26 [Test]:  100%|████████| 79/79 [00:08<00:00, 9.8batch/s, Loss=0.5876, Acc=0.6892]

Tóm tắt Epoch 1:
   Train Loss: 0.6234 | Train Acc: 0.6543
   Test Loss:  0.5876 | Test Acc:  0.6892
   Thời gian: 53.2s
Đã lưu mô hình tốt nhất mới!
```

#### Các File được Tạo ra:

```
QITSA/
├── Loss_CR.png              # Đường cong loss huấn luyện
├── Acc_CR.png               # Đường cong độ chính xác
├── best_model_CR.pt         # Checkpoint mô hình tốt nhất
└── training_log.txt         # Log huấn luyện chi tiết
```

### Ước tính Thời gian Huấn luyện

| Phần cứng | Bộ dữ liệu CR | Bộ dữ liệu MR | Bộ dữ liệu SST |
|----------|------------|------------|-------------|
| **NVIDIA RTX 4090** | 3-5 phút | 6-8 phút | 20-25 phút |
| **NVIDIA RTX 3080** | 5-7 phút | 8-12 phút | 25-35 phút |
| **NVIDIA GTX 1080 Ti** | 8-12 phút | 15-20 phút | 45-60 phút |
| **Chỉ CPU (16 lõi)** | 30-45 phút | 60-90 phút | 4-6 giờ |
| **Chỉ CPU (8 lõi)** | 45-60 phút | 90-120 phút | 6-10 giờ |

### Giám sát Tiến trình Huấn luyện

#### Giám sát Thời gian Thực:
```bash
# Giám sát việc sử dụng GPU:
nvidia-smi -l 1

# Theo dõi log huấn luyện:
tail -f training_log.txt

# Kiểm tra các biểu đồ:
ls -la *.png
```

#### Các Chỉ số Chính:
1. **Training Loss**: Nên giảm đều đặn
2. **Test Loss**: Nên giảm nhưng có thể dao động
3. **Training Accuracy**: Nên tăng hướng tới 0.85-0.95
4. **Test Accuracy**: Nên gần với training accuracy
5. **Bộ nhớ GPU**: Nên giữ ổn định

## Tìm hiểu Sâu về Kiến trúc Mô hình

### Triết lý Thiết kế Lượng tử

QITSA cách mạng hóa phân tích cảm xúc bằng cách xử lý văn bản như hệ lượng tử.

#### So sánh Phương pháp

**Truyền thống:**
```python
# Từ như vector đơn giản
word_embedding = nn.Embedding(vocab_size, embedding_dim)
word_vectors = word_embedding(word_indices)
hidden_state = lstm(word_vectors)
sentiment_score = classifier(hidden_state)
```

**QITSA (Lượng tử):**
```python
# Từ như trạng thái lượng tử phức
real_embedding = nn.Embedding(vocab_size, embedding_dim)
imag_embedding = nn.Embedding(vocab_size, embedding_dim)

# Tạo trạng thái chồng chập lượng tử
real_states = real_embedding(word_indices)
imag_states = imag_embedding(word_indices)
quantum_state = complex(real_states, imag_states)

# Áp dụng phép biến đổi lượng tử
quantum_features = quantum_projection(quantum_state)
sentiment_score = quantum_measurement(quantum_features)
```

### Các Thành phần Lượng tử Cốt lõi

#### 1. Biểu diễn Trạng thái Lượng tử

Mỗi từ được biểu diễn như trạng thái lượng tử với:
- **Thành phần Thực**: Ý nghĩa ngữ nghĩa tường minh
- **Thành phần Ảo**: Thông tin cảm xúc/ngữ cảnh ngầm định
- **Chồng chập Lượng tử**: Từ có thể tồn tại trong nhiều trạng thái cảm xúc

#### 2. Kiến trúc projection_Euler

Trái tim của xử lý lượng tử QITSA:

```python
class projection_Euler(nn.Module):
    """
    Triển khai các phép biến đổi lượng tử sử dụng công thức Euler:
    e^(iθ) = cos(θ) + i*sin(θ)
    
    Áp dụng các phép quay lượng tử vào embedding từ,
    cho phép hiệu ứng chồng chập và giao thoa lượng tử.
    """
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, num_layers):
        super(projection_Euler, self).__init__()
        
        # Embedding tiền huấn luyện cho cả thực và ảo
        self.real_embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.imag_embedding = nn.Embedding.from_pretrained(embedding_matrix.clone())
        
        # Các lớp biến đổi lượng tử
        self.quantum_layers = nn.ModuleList([
            QuantumTransformLayer(embedding_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Lớp phân loại cuối cùng
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
```

#### 3. Các Phép toán Lượng tử

**A. Chồng chập Lượng tử**
- Từ có thể biểu diễn nhiều trạng thái cảm xúc đồng thời
- Xử lý cảm xúc mơ hồ hoặc phụ thuộc ngữ cảnh
- Ví dụ: "not bad" vừa tiêu cực (có "not") vừa tích cực (ý nghĩa tổng thể)

**B. Giao thoa Lượng tử**
- Giao thoa tạo dựng khuếch đại tín hiệu cảm xúc nhất quán
- Giao thoa triệt tiêu triệt bỏ thông tin cảm xúc mâu thuẫn
- Giải quyết các chỉ báo cảm xúc mâu thuẫn

**C. Đo lường Lượng tử**
- Sụp đổ chồng chập lượng tử thành dự đoán cảm xúc cổ điển
- Giữ nguyên tính không chắc chắn lượng tử đến quyết định cuối cùng
- Cho phép diễn giải xác suất của cảm xúc

### Sơ đồ Luồng Kiến trúc

```
Câu đầu vào: "This movie was not very good"
       ↓
Phân đoạn từ: ["this", "movie", "was", "not", "very", "good"]
       ↓
Chuyển thành chỉ số: [45, 123, 67, 892, 34, 156]
       ↓
Embedding Lượng tử:
├─ Thành phần Thực: [ý nghĩa ngữ nghĩa]
└─ Thành phần Ảo: [ngữ cảnh cảm xúc]
       ↓
Các lớp Biến đổi Lượng tử (×2):
├─ Quay Pha 1: Áp dụng quay lượng tử
├─ Điều chế Cảm xúc: Tích hợp điểm SentiWordNet
└─ Quay Pha 2: Biến đổi lượng tử tiếp theo
       ↓
Đo lường Lượng tử: |ψ|² = |thực|² + |ảo|²
       ↓
Trích xuất Đặc trưng CNN (tùy chọn):
├─ Conv1D(kernel_size=3): trigrams
├─ Conv1D(kernel_size=4): 4-grams
└─ Conv1D(kernel_size=5): 5-grams
       ↓
Max Pooling: Trích xuất các đặc trưng quan trọng nhất
       ↓
Lớp Phân loại: Linear → Sigmoid
       ↓
Điểm Cảm xúc: 0.23 (cảm xúc tiêu cực)
```

### Ưu điểm Chính của Kiến trúc Lượng tử

**Khả năng Biểu diễn Nâng cao**
- Số phức cung cấp biểu diễn phong phú hơn chỉ số thực
- Mô hình hóa trạng thái cảm xúc mơ hồ tốt hơn
- Xử lý dịch chuyển cảm xúc phụ thuộc ngữ cảnh tốt hơn

**Khả năng Diễn giải Cải thiện**
- Trực quan hóa trạng thái lượng tử cho thấy cảm xúc tiến hóa qua mạng
- Thông tin pha tiết lộ các mẫu cảm xúc tinh tế
- Các mẫu biên độ làm nổi bật từ mang cảm xúc quan trọng

**Bền vững với Phủ định và Mâu thuẫn**
- Giao thoa lượng tử tự nhiên xử lý phủ định ("not good" vs "good")
- Chồng chập cho phép mô hình hóa cảm xúc hỗn hợp ("good but expensive")
- Hiệu suất tốt hơn trên cấu trúc ngôn ngữ phức tạp

**Nền tảng Lý thuyết**
- Dựa trên các nguyên lý cơ học lượng tử đã được thiết lập
- Cung cấp framework toán học để hiểu hành vi mô hình
- Mở ra hướng nghiên cứu mới trong NLP lấy cảm hứng từ lượng tử

## Quy trình Xử lý Dữ liệu Đầy đủ

Quy trình xử lý dữ liệu QITSA chuyển đổi văn bản thô thành tensor sẵn sàng cho lượng tử qua 5 giai đoạn chính.

### Tổng quan Quy trình

```
File TSV Thô → Câu được Phân đoạn → Chỉ số Từ → Điểm Cảm xúc → Tensor theo Batch → Đầu vào Mô hình
```

### Giai đoạn 1: Xử lý Câu (get_sentences)

**Mục đích**: Phân tích file TSV và trích xuất câu sạch với nhãn  
**Đầu vào**: Đường dẫn file TSV  
**Đầu ra**: Danh sách tuple `[(danh_sách_từ, độ_dài_câu, nhãn), ...]`

**Các bước Xử lý Chính:**
1. **Đọc File**: Mở TSV với mã hóa UTF-8
2. **Làm sạch Dòng**: Loại bỏ khoảng trắng và xử lý dòng trống
3. **Tách Tab**: Tách văn bản khỏi nhãn sử dụng ký tự tab
4. **Phân đoạn từ**: Tách từ đơn giản dựa trên khoảng trắng
5. **Xác thực Nhãn**: Đảm bảo nhãn là giá trị nhị phân hợp lệ (0.0 hoặc 1.0)
6. **Xử lý Lỗi**: Xử lý các dòng không đúng định dạng với cảnh báo

### Giai đoạn 2: Ánh xạ Từ-sang-Chỉ số (lookup_table)

**Mục đích**: Chuyển đổi danh sách từ thành chỉ số số để xử lý mạng nơ-ron  
**Đầu vào**: Các câu từ get_sentences(), từ điển từ-sang-chỉ số  
**Đầu ra**: Các câu với chỉ số từ thay vì từ

**Xử lý Từ vựng:**
- **Từ Đã biết**: Ánh xạ đến chỉ số embedding tiền huấn luyện
- **Từ Không xác định (UNK)**: Gán chỉ số 1 (token không xác định toàn cục)
- **Token Đệm**: Chỉ số 0 (dùng để đệm câu)
- **Thống kê**: Theo dõi tỷ lệ từ không xác định để đánh giá chất lượng

### Giai đoạn 3: Nâng cao Điểm Cảm xúc (get_batch_word_sentiment_scores)

**Mục đích**: Sử dụng SentiWordNet để gán điểm cực tính cảm xúc cho mỗi từ  
**Nâng cao**: Thêm thông tin cảm xúc lấy cảm hứng từ lượng tử vào biểu diễn từ

```python
from nltk.corpus import sentiwordnet as swn
from tqdm import tqdm

def get_word_sentiment_score(word):
    """
    Tính toán điểm cảm xúc ròng của một từ từ SentiWordNet
    Trả về giá trị trong [-1, 1] với -1 là tiêu cực nhất, +1 là tích cực nhất
    """
    synsets = list(swn.senti_synsets(word))
    
    if not synsets:
        return 0.0  # Trung lập cho từ không xác định
    
    # Tính điểm trung bình qua tất cả các nghĩa
    total_pos = sum(s.pos_score() for s in synsets)
    total_neg = sum(s.neg_score() for s in synsets)
    
    # Điểm ròng: tích cực - tiêu cực
    net_sentiment = (total_pos - total_neg) / len(synsets)
    return net_sentiment

def get_batch_word_sentiment_scores(sentences):
    """
    Tính toán điểm cảm xúc cho tất cả từ trong batch các câu
    Hiển thị thanh tiến trình để theo dõi
    """
    sentence_sentiment_scores = []
    
    for sentence, length, label in tqdm(sentences, desc="Tính điểm cảm xúc"):
        word_scores = []
        for word in sentence:
            score = get_word_sentiment_score(word)
            word_scores.append(score)
        sentence_sentiment_scores.append(word_scores)
    
    return sentence_sentiment_scores
```

### Giai đoạn 4: Đệm Batch (pad_sentence_plus)

**Mục đích**: Đảm bảo tất cả câu trong một batch có độ dài bằng nhau để xử lý GPU hiệu quả

**Chiến lược Đệm:**
- Tìm độ dài câu tối đa trong batch
- Đệm các câu ngắn hơn với PAD_IDX (0)
- Đệm điểm cảm xúc với 0.0 (trung lập)
- Duy trì cấu trúc batch cho xử lý song song

### Giai đoạn 5: Tạo Iterator Batch

**Mục đích**: Tạo các batch dữ liệu cho huấn luyện/kiểm tra hiệu quả

**Tạo Batch:**
- Nhóm các mẫu theo BATCH_SIZE
- Trộn dữ liệu huấn luyện (tùy chọn)
- Tạo iterator PyTorch cho việc lặp hiệu quả

## Khắc phục Sự cố Thường gặp

### Lỗi CUDA Out of Memory

**Triệu chứng:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MB
```

**Giải pháp:**
```python
# 1. Giảm kích thước batch
BATCH_SIZE = 8  # hoặc thậm chí 4

# 2. Giảm kích thước mô hình
HIDDEN_DIM = 2
num_layers = 1

# 3. Xóa cache CUDA
import torch
torch.cuda.empty_cache()

# 4. Sử dụng gradient checkpointing (nâng cao)
```

### Lỗi Dữ liệu NLTK Thiếu

**Triệu chứng:**
```
LookupError: Resource 'wordnet' not found
```

**Giải pháp:**
```python
import nltk
import ssl

# Bỏ qua xác minh SSL nếu cần
ssl._create_default_https_context = ssl._create_unverified_context

# Tải dữ liệu bị thiếu
nltk.download('wordnet')
nltk.download('sentiwordnet')
```

### Vấn đề Không Tìm thấy Embedding

**Triệu chứng:**
```
FileNotFoundError: [Errno 2] No such file or directory: './sub_word_vector/CR_word_vector.txt'
```

**Giải pháp:**
- Đảm bảo bạn đã tải và giải nén các file embedding được tiền huấn luyện
- Kiểm tra cấu trúc thư mục khớp với đường dẫn dự kiến
- Xác minh DATA_SET được đặt chính xác

### Huấn luyện Chậm trên CPU

**Triệu chứng:** Huấn luyện mất nhiều giờ

**Giải pháp:**
```python
# 1. Giảm epoch để kiểm tra nhanh
N_EPOCHS = 5

# 2. Sử dụng bộ dữ liệu nhỏ hơn (CR)
DATA_SET = 'CR'

# 3. Giảm kích thước mô hình
HIDDEN_DIM = 2
num_layers = 1

# 4. Xem xét sử dụng Google Colab (GPU miễn phí)
```

## Hiểu Kết quả Huấn luyện

### Diễn giải Metrics

**Training Loss:**
- Nên giảm đều đặn qua các epoch
- Dao động = tốc độ học có thể quá cao
- Không giảm = có thể cần nhiều epoch hoặc tốc độ học khác

**Test Loss:**
- Nên giảm nhưng có thể dao động
- Nếu tăng trong khi training loss giảm = overfitting
- Gap lớn với training loss = cần regularization

**Accuracy:**
- Độ chính xác cuối cùng phải > 80% cho hầu hết bộ dữ liệu
- Training và test accuracy nên gần nhau
- Gap lớn = overfitting

### Expected Results

**Độ chính xác Điển hình:**
- **CR Dataset**: 83-86%
- **MR Dataset**: 80-84%
- **SST Dataset**: 82-85%
- **SUBJ Dataset**: 91-93%
- **MPQA Dataset**: 85-88%
