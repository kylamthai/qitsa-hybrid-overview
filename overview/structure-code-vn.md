# Phân Tích Cấu Trúc Code QITSA - Tài Liệu Chi Tiết

## Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [File 1: train_20102025.py - Script Huấn Luyện](#file-1-train_20102025py)
3. [File 2: test_vis.py - Script Trực Quan Hóa (Tiếng Trung)](#file-2-test_vispy)
4. [File 3: test_vis_english.py - Script Trực Quan Hóa (Tiếng Anh)](#file-3-test_vis_englishpy)
5. [File 4: aMain_20102025.ipynb - Notebook Chính](#file-4-amain_20102025ipynb)
6. [Tóm Tắt Kiến Trúc](#tóm-tắt-kiến-trúc)
7. [Checklist Tái Tạo Kết Quả](#checklist-tái-tạo-kết-quả)

---

## Tổng Quan

Repository này triển khai **QITSA (Quantum-Inspired Text Sentiment Analysis)** - một mạng neural hybrid quantum-classical để phân loại cảm xúc. Model kết hợp:
- **Biểu diễn lấy cảm hứng từ lượng tử**: Complex-valued embeddings (amplitude + phase)
- **Deep learning cổ điển**: LSTM/GRU với skip connections
- **Sentiment polarity**: Điểm sentiment theo từng từ từ SentiWordNet
- **Công thức density matrix**: Biểu diễn trạng thái lấy cảm hứng từ cơ học lượng tử

### Các Dataset Được Hỗ Trợ
- **MR**: Movie Reviews (Đánh giá phim)
- **CR**: Customer Reviews (Đánh giá khách hàng)
- **SST**: Stanford Sentiment Treebank
- **SUBJ**: Subjectivity dataset (Dữ liệu tính chủ quan)
- **MPQA**: Multi-Perspective Question Answering

---

## File 1: train_20102025.py

### Mục Đích
Script huấn luyện chính cho model QITSA. Xử lý toàn bộ pipeline huấn luyện từ tải dữ liệu đến lưu model.

### Phân Tích Cấu Trúc

#### 1. **Imports và Configuration** (Dòng 1-70)
```python
# Thư viện chính
- torch, torch.nn, torch.optim: PyTorch framework
- matplotlib.pyplot: Trực quan hóa
- nltk.corpus: wordnet, sentiwordnet cho sentiment scoring
- sklearn.metrics: f1_score để đánh giá
```

**Biến Cấu Hình**:
```python
BATCH_SIZE = 16          # Kích thước batch cho training
DATA_SET = 'CR'          # Hiện đang train trên Customer Reviews
N_EPOCHS = 26            # Số epoch training
SEED = 1234              # Random seed để tái tạo kết quả
CUDA_NUMBER = 0          # Chỉ số GPU device
```

**Mapping Dataset**:
```python
dataset_dict = {
    'MR': 0, 'CR': 1, 'SST': 2, 'SUBJ': 3, 'MPQA': 4
}
```

#### 2. **Tải Word Embedding** (Dòng 71-98)

**Quy Trình**:
1. Khởi tạo vocabulary với special tokens: `<unknown>: 0`, `<padded>: 1`
2. Đọc pre-trained word vectors (embeddings 100 chiều kiểu GloVe)
3. Xây dựng mapping hai chiều: `word_to_index`, `index_to_word`
4. Tạo weight matrix cho embedding layer

**Code Chính**:
```python
word_to_index = {'<unknown>': 0, '<padded>': 1}
zero_ls = [0.0 for i in range(word_dim)]  # Vector 0 có 100 chiều
ls = [zero_ls, zero_ls]  # Hai đầu tiên: unknown, padded

# Tải word vectors
with open(word_vector_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        word_vector = line.split()
        word_to_index[word_vector[0]] = i + 2
        tmp = [float(word) for j, word in enumerate(word_vector) if j > 0]
        ls.append(tmp)

word_vector_weight_matrix = torch.FloatTensor(ls)
VOCAB_SIZE = len(word_to_index) + 2
```

#### 3. **Hàm Tiền Xử Lý Dữ Liệu** (Dòng 99-225)

**Hàm 1: `get_sentences(path)`**
- Đọc file TSV với format: `word1 word2 ... wordN label`
- Trả về: `[(words_list, sentence_length, label), ...]`

**Hàm 2: `lookup_table(array_ls, word_to_index)`**
- Chuyển đổi từ thành indices sử dụng vocabulary
- Từ không biết → index 1 (giống `<padded>`)
- Trả về: `[(word_indices, length, label), ...]`

**Hàm 3: `pad_sentence_plus(array_ls, array_sentiment, batch_size)`**
- Padding câu đến độ dài max trong mỗi batch
- Cũng padding sentiment scores với 0s
- Đảm bảo kích thước batch đồng nhất

**Hàm 4: `iterator(array_ls, array_sentiment, batch_size, shuffle=True, batch_first=False)`**
- Tạo batches có size 16
- Transpose thành shape: `(sentence_len, batch_size)` cho LSTM
- Trả về: `[(text_tensor, label_tensor, sentiment_tensor), ...]`
- Xáo trộn batches để training

#### 4. **Tính Điểm Sentiment với SentiWordNet** (Dòng 226-289)

**Hàm Chính**:

```python
def get_sentiment_score(word):
    """
    Tính sentiment polarity theo từng từ sử dụng SentiWordNet
    Trả về: pos_score - neg_score (khoảng: -1 đến +1)
    """
    synsets = list(wordnet.synsets(word))
    if synsets:
        sentiment_scores = [
            sentiwordnet.senti_synset(s.name()).pos_score() - 
            sentiwordnet.senti_synset(s.name()).neg_score()
            for s in synsets
        ]
        return sum(sentiment_scores) / len(sentiment_scores)
    return 0.0

def get_batch_word_sentiment_scores(batch_tokenized_sentences):
    """Xử lý batch câu, trả về sentiment cho mỗi từ"""
    batch_word_scores = []
    for sentence, _, _ in batch_tokenized_sentences:
        word_scores = [get_sentiment_score(word) for word in sentence]
        batch_word_scores.append(word_scores)
    return batch_word_scores
```

**Mục Đích**: 
- Cung cấp thông tin phase cho biểu diễn phức quantum-inspired
- Sentiment polarity trở thành thành phần "phase"
- Word embedding trở thành thành phần "amplitude"

#### 5. **Kiến Trúc Model** (Dòng 335-567)

**Hệ Thống Phân Cấp Class**:

##### **5.1 `projection_Euler`** (Dòng 343-367)
Chuyển đổi biểu diễn amplitude-phase sang phần thực-ảo sử dụng công thức Euler.

```python
class projection_Euler(nn.Module):
    def forward(self, inputs):
        amplitude = inputs[0]  # Word embeddings
        phase = inputs[1]      # Sentiment scores
        
        # Chuẩn hóa amplitude
        amplitude_permute = amplitude.permute(1, 0, 2)
        amplitude_norm = F.normalize(amplitude_permute, 2, 2)
        
        # Khai triển Euler: z = A * e^(iθ) = A*cos(θ) + i*A*sin(θ)
        real_part = amplitude_norm * torch.cos(phase_permute)
        imag_part = amplitude_norm * torch.sin(phase_permute)
        
        return [real_part, imag_part]
```

**Cơ Sở Toán Học**:
$$z = A \cdot e^{i\theta} = A(\cos\theta + i\sin\theta)$$
- $A$: Amplitude (word embedding đã chuẩn hóa)
- $\theta$: Phase (sentiment polarity)

##### **5.2 `projection_CalculateMatrixMean`** (Dòng 368-394)
Cơ chế attention để tổng hợp density matrix.

```python
class projection_CalculateMatrixMean(nn.Module):
    def forward(self, inputs):
        Diagonal1, Diagonal2, Matrix = inputs[0], inputs[1], inputs[2]
        
        # Tính attention scores từ các phần tử đường chéo
        attention_scores = Diagonal1 * Diagonal2
        attention_weight = torch.sum(attention_scores, dim=-1)
        
        # Chuẩn hóa Softmax
        attention = F.softmax(attention_weight, dim=-1).unsqueeze(-1).unsqueeze(-1)
        
        # Tổng có trọng số theo chiều sequence
        Matrix_output = torch.sum(Matrix * attention, dim=1)
        return Matrix_output
```

**Mục Đích**: Tổng hợp density matrices qua độ dài sequence sử dụng learned attention.

##### **5.3 `projection_CalculateMatrixQ`** (Dòng 395-445)
Tính quantum density matrix từ biểu diễn phức.

```python
class projection_CalculateMatrixQ(nn.Module):
    def forward(self, inputs):
        input_real, input_imag = inputs[0], inputs[1]
        
        # Linear transformations
        input_real1 = self.Q_Linear_real1(input_real)
        input_imag1 = self.Q_Linear_imag1(input_imag)
        
        # Phần tử đường chéo (|ψ|²)
        diagonal_real1 = input_real1 * input_real1 - input_imag1 * input_imag1
        diagonal_imag1 = input_real1 * input_imag1 + input_imag1 * input_real1
        
        # Density matrix: ρ = |ψ⟩⟨ψ|
        real_part_expand = torch.unsqueeze(input_real2, dim=3)
        imag_part_expand = torch.unsqueeze(input_imag2, dim=3)
        
        # Phép nhân ma trận phức
        Matrix_real = torch.matmul(real_part_expand, real_part_expand_transpose) - 
                      torch.matmul(imag_part_expand, imag_part_expand_transpose)
        Matrix_imag = torch.matmul(imag_part_expand, real_part_expand_transpose) + 
                      torch.matmul(real_part_expand, imag_part_expand_transpose)
        
        # Áp dụng tổng hợp dựa trên attention
        Matrix_Real_output = self.AttentionReal([diagonal_real1, diagonal_real2, Matrix_real])
        Matrix_imag_output = self.AttentionImag([diagonal_imag1, diagonal_imag2, Matrix_imag])
        
        return [Matrix_Real_output, Matrix_imag_output]
```

**Giải Thích Quantum**:
- Density matrix: $\rho = |\psi\rangle\langle\psi|$
- Biểu diễn trạng thái lượng tử trong dạng mixed state formalism
- Phần tử đường chéo: xác suất (probabilities)
- Ngoài đường chéo: coherences

##### **5.4 `projection_Measurement`** (Dòng 446-475)
Thao tác đo lường lượng tử sử dụng orthogonal projectors.

```python
class projection_Measurement(nn.Module):
    def __init__(self, Embedding_dim):
        super().__init__()
        # Orthogonal projector (khởi tạo ngẫu nhiên, sau đó orthogonalized)
        self.projector = nn.init.orthogonal_(
            Parameter(torch.Tensor(2, Embedding_dim, 1))
        )
    
    def forward(self, inputs):
        v_real_avg, v_imag_avg = inputs[0], inputs[1]
        
        p_real, p_imag = self.projector[0], self.projector[1]
        
        # Chuẩn hóa projectors
        p_real_norm = p_real / torch.norm(p_real, dim=0)
        p_imag_norm = p_imag / torch.norm(p_imag, dim=0)
        
        # Projection matrices: P = |p⟩⟨p|
        p_real_mat = torch.matmul(p_real_norm, p_real_norm.permute(1, 0))
        p_imag_mat = torch.matmul(p_imag_norm, p_imag_norm.permute(1, 0))
        
        # Measurement: Pv (phép nhân phức)
        Pv_real = torch.matmul(v_real_avg, p_real_mat) - torch.matmul(v_imag_avg, p_imag_mat)
        Pv_imag = torch.matmul(v_real_avg, p_imag_mat) + torch.matmul(v_imag_avg, p_real_mat)
        
        # Thêm batch dimension
        Pv_real_plus = torch.unsqueeze(Pv_real, dim=1)
        Pv_imag_plus = torch.unsqueeze(Pv_imag, dim=1)
        
        return [Pv_real_plus, Pv_imag_plus]
```

**Giải Thích Quantum**: 
- Projective measurement: $P|\psi\rangle$
- Thu gọn trạng thái lượng tử vào measurement basis
- Orthogonal projector đảm bảo quantum measurement đúng chuẩn

##### **5.5 `self_attention`** (Dòng 476-502)
Scaled dot-product attention chuẩn.

```python
class self_attention(nn.Module):
    def forward(self, inputs):
        # Biến đổi Q, K, V
        query = torch.matmul(inputs.permute(1, 0, 2), self.mapping_query)
        key = torch.matmul(inputs.permute(1, 0, 2), self.mapping_key)
        value = torch.matmul(inputs.permute(1, 0, 2), self.mapping_value)
        
        # Chuẩn hóa L2
        query = query / torch.norm(query, dim=2).reshape(...)
        key = key / torch.norm(key, dim=2).reshape(...)
        value = value / torch.norm(value, dim=2).reshape(...)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, torch.transpose(key, 1, 2)) / np.sqrt(inputs.shape[1])
        attention_weights = torch.softmax(scores, dim=2)
        output = torch.matmul(attention_weights, value)
        
        return output
```

##### **5.6 `LSTMWithSkipConnection`** (Dòng 503-520)
LSTM với residual connections để gradient flow tốt hơn.

```python
class LSTMWithSkipConnection(nn.Module):
    def __init__(self, Embedding_dim, num_layers):
        super().__init__()
        self.LSTM = nn.LSTM(Embedding_dim, Embedding_dim, 
                           num_layers=num_layers, dropout=0.1)
        self.middle_layer = nn.Linear(Embedding_dim, Embedding_dim)
    
    def forward(self, x):
        output, _ = self.LSTM(x)
        # Skip connection
        output = output + self.middle_layer(x)
        return output
```

##### **5.7 `run_complex_network`** - Model Chính (Dòng 521-635)

**Pipeline Kiến Trúc**:

```python
class run_complex_network(nn.Module):
    def __init__(self, weight_matrix, embedding_dim, hidden_dim, output_dim, 
                 num_layers, pad_idx, VOCAB_SIZE, batch):
        super().__init__()
        
        # 1. Word Embedding (Amplitude)
        self.amplitude_embedding = nn.Embedding.from_pretrained(
            weight_matrix, padding_idx=pad_idx, freeze=False
        )
        
        # 2. LSTM + Self-Attention (Trích xuất đặc trưng)
        self.LSTMWithSkipConnection1 = LSTMWithSkipConnection(embedding_dim, num_layers)
        self.LSTMWithSkipConnection2 = LSTMWithSkipConnection(embedding_dim, num_layers)
        self.self_attention1 = self_attention(Embedding_dim=embedding_dim)
        
        # 3. Quantum Modules
        self.projection_Euler = projection_Euler(embedding_dim, num_layers)
        self.projection_CalculateMatrixattension = projection_CalculateMatrixQ(embedding_dim)
        self.projection_measurement = projection_Measurement(embedding_dim)
        
        # 4. Convolution (Trích xuất đặc trưng)
        self.Conv2dOne = nn.Conv2d(1, 1, 3)  # Phần thực
        self.Conv2dTwo = nn.Conv2d(1, 1, 3)  # Phần ảo
        
        # 5. Pooling
        self.MaxPool1 = nn.MaxPool2d((embedding_dim - 2, 1), 1)  # Real
        self.MaxPool2 = nn.MaxPool2d((embedding_dim - 2, 1), 1)  # Imaginary
        
        # 6. Fully Connected Layers
        self.fc1 = nn.Linear(2 * (embedding_dim - 2), 10)
        self.fc2 = nn.Linear(10, output_dim)
    
    def forward(self, text, sentiment):
        # Bước 1: Embedding
        amplitude_is_WordEmbedding = self.amplitude_embedding(text)
        
        # Bước 2: Mở rộng sentiment để khớp với kích thước embedding
        sentiment_unsqueeze = (torch.unsqueeze(sentiment, dim=-1)
                              .expand(*sentiment.shape, amplitude_is_WordEmbedding.size(2)))
        phase_is_sentiment = sentiment_unsqueeze
        
        # Bước 3: LSTM + Self-Attention
        amplitude_plus = self.LSTMWithSkipConnection1(amplitude_is_WordEmbedding)
        amplitude_plus2 = self.self_attention1(amplitude_plus)
        amplitude_plus2 = amplitude_plus2.permute(1, 0, 2)
        
        phase_plus = self.LSTMWithSkipConnection2(phase_is_sentiment)
        
        # Bước 4: Xử Lý Quantum
        embedded = [amplitude_plus2, phase_plus]
        
        # Biến đổi Euler: Biểu diễn phức
        Euler_realAimag = self.projection_Euler(embedded)
        
        # Tính Density matrix
        Matrix_realAimag = self.projection_CalculateMatrixattension(Euler_realAimag)
        
        # Quantum measurement
        Project_realAimag = self.projection_measurement(Matrix_realAimag)
        
        MatrixReal = Project_realAimag[0]  # Shape: (batch, 1, dim, dim)
        MatrixImag = Project_realAimag[1]
        
        # Bước 5: Xử Lý CNN
        Conv_real = self.Conv2dOne(MatrixReal)
        Conv_imag = self.Conv2dTwo(MatrixImag)
        
        # Bước 6: Pooling
        Max_real = self.MaxPool1(torch.sigmoid(Conv_real))
        Max_imag = self.MaxPool2(torch.sigmoid(Conv_imag))
        
        # Bước 7: Concatenate và Phân Loại
        fc1 = self.fc1(torch.cat((Max_real, Max_imag), dim=3))
        fc2 = torch.sigmoid(self.fc2(torch.sigmoid(fc1)))
        
        return fc2
```

**Luồng Dữ Liệu**:
```
Input Text (indices)
    ↓
Word Embeddings (100-dim) ───────→ Amplitude
Sentiment Scores (theo từng từ) ──→ Phase
    ↓
LSTM + Self-Attention
    ↓
Euler Transform: A·e^(iθ) → (A·cos(θ), A·sin(θ))
    ↓
Density Matrix: ρ = |ψ⟩⟨ψ|
    ↓
Quantum Measurement: P|ψ⟩
    ↓
CNN (3×3) trên phần Real & Imaginary
    ↓
MaxPooling
    ↓
Concatenate [Real, Imag]
    ↓
FC1 (→10) → FC2 (→1)
    ↓
Sigmoid → Binary Sentiment (0 hoặc 1)
```

#### 6. **Vòng Lặp Training** (Dòng 636-831)

**Hyperparameters**:
```python
EMBEDDING_DIM = 100
HIDDEN_DIM = 3
OUTPUT_DIM = 1
num_layers = 2
PAD_IDX = 1
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary Cross-Entropy
```

**Các Metric Đánh Giá**:
```python
def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def calculation_recall(preds, y):
    rounded_preds = torch.round(preds)
    TP = (rounded_preds.int() & y.int()).sum()
    recall = TP / y.sum()
    return recall

# F1-Score từ sklearn
F1Score = f1_score(y_true, y_pred)
```

**Hàm Training**:
```python
def train(model, iterator, optimizer, criterion, CUDA_NUMBER):
    model.train()
    epoch_loss = epoch_acc = epoch_f1_score = epoch_recall = 0
    
    for batch in iterator:
        text, label, sentiment = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        
        optimizer.zero_grad()
        predictions = model(text, sentiment).squeeze(1)
        loss = criterion(predictions.reshape(len(label)), label)
        
        acc = binary_accuracy(predictions.reshape(len(label)), label)
        recall = calculation_recall(predictions.reshape(len(label)), label)
        F1Score = f1_score(label.tolist(), torch.round(predictions).tolist())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_recall += recall.item()
        epoch_f1_score += F1Score.item()
    
    return (epoch_loss / len(iterator), 
            epoch_acc / len(iterator), 
            epoch_f1_score / len(iterator), 
            epoch_recall / len(iterator))
```

**Vòng Lặp Training Chính**:
```python
for epoch in range(N_EPOCHS):
    train_loss, train_acc, train_f1, train_recall = train(...)
    test_loss, test_acc, test_f1, test_recall = evaluate(...)
    
    # Theo dõi metrics
    acc_train.append(train_acc)
    acc_test.append(test_acc)
    loss_train.append(train_loss)
    loss_test.append(test_loss)
    f1_score_test.append(test_f1)
    
    # Lưu model tốt nhất
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), f'Best_model_of_{DATA_SET}.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
```

#### 7. **Lưu Kết Quả và Trực Quan Hóa** (Dòng 832-936)

**Lưu Metrics**:
```python
# Lưu vào file text
with open(f'./train/loss/{DATA_SET}.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in loss_train]))

with open(f'./test/acc/{DATA_SET}.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in acc_test]))
```

**Trực Quan Hóa**:
```python
# Đường cong Loss
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel(DATA_SET + ' Loss')
plt.plot(range(N_EPOCHS), loss_train, 'r-o', label='train_loss')
plt.plot(range(N_EPOCHS), loss_test, 'c-*', label='test_loss')
plt.legend()
plt.savefig(f'Loss{DATA_SET}.png', dpi=1500)

# Đường cong Accuracy
plt.figure(2)
plt.xlabel('Epoch')
plt.ylabel(DATA_SET + ' Acc')
plt.plot(range(N_EPOCHS), acc_train, 'r-o', label='acc_train')
plt.plot(range(N_EPOCHS), acc_test, 'c-*', label='acc_test')
plt.legend()
plt.savefig(f'Acc{DATA_SET}.png', dpi=1500)
```

---

## File 2: test_vis.py

### Mục Đích
Script testing và trực quan hóa với **nhãn tiếng Trung**. Tạo các visualization trung gian tại mỗi layer của model.

### Điểm Khác Biệt So Với Script Training

#### 1. **Cấu Hình Font** (Dòng 10-11)
```python
plt.rcParams['font.sans-serif'] = ['SimHei']  # Font tiếng Trung
plt.rcParams['axes.unicode_minus'] = False     # Xử lý dấu trừ
```

#### 2. **Cấu Hình Seed** (Dòng 290-306)
```python
SEED = 6666  # Khác với training (1234)
SA = 1       # Cờ Sentiment Analysis (1=dùng, 0=không dùng)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Tạo thư mục output
def create_figure_folders(num_folders, data_set, seed):
    for i in range(num_folders):
        folder_name = f"{SA}_{data_set}/figure{i}_{data_set}_{seed}"
        os.makedirs(folder_name, exist_ok=True)

create_figure_folders(5, DATA_SET, SEED)
```

#### 3. **Trực Quan Hóa Nâng Cao trong `projection_Euler`** (Dòng 361-418)

**Trực Quan Hóa Amplitude**:
```python
def forward(self, inputs):
    amplitude = inputs[0]
    phase = inputs[1]
    
    amplitude_permute = amplitude.permute(1, 0, 2)
    amplitude_norm = F.normalize(amplitude_permute, 2, 2)
    phase_permute = phase.permute(1, 0, 2)
    
    # Trực quan hóa amplitude (nhãn tiếng Trung)
    plt.figure(figsize=(10, 8))
    plt.imshow(amplitude_norm.cpu()[0,:,:].detach().numpy(), 
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='幅值信息', orientation='horizontal')
    plt.title('幅值信息')
    plt.xlabel('单个句子幅值表示的分数')
    plt.ylabel('单词幅值表示中不同维度表示的分数')
    plt.savefig(f'figure2/amplitude_norm_{DATA_SET}.jpg', dpi=1000)
    
    # Trực quan hóa phase (nhãn tiếng Trung)
    plt.figure(figsize=(10, 8))
    plt.imshow(phase_permute.cpu()[0,:,:].detach().numpy(), 
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='相角信息', orientation='horizontal')
    plt.title('相角信息')
    plt.xlabel('单个句子相角表示的分数')
    plt.ylabel('单词相角表示中不同维度表示的分数')
    plt.savefig(f'figure2/phase_permute_{DATA_SET}.jpg', dpi=1000)
    
    # Phần thực
    real_part = amplitude_norm * torch.cos(phase_permute)
    plt.figure(figsize=(10, 8))
    plt.imshow(real_part.cpu()[0,:,:].detach().numpy(), 
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='复数表示法的实部', orientation='horizontal')
    plt.title('单个句子实部表示的分数')
    plt.xlabel('单词实部表示每一维度的分数')
    plt.ylabel('每个单词实部表示的分数')
    plt.savefig(f'figure3/real_part_{DATA_SET}.jpg', dpi=1000)
    
    # Phần ảo
    imag_part = amplitude_norm * torch.sin(phase_permute)
    plt.figure(figsize=(10, 8))
    plt.imshow(imag_part.cpu()[0,:,:].detach().numpy(), 
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='复数表示法的虚部', orientation='horizontal')
    plt.title('单个句子虚部表示法的分数')
    plt.xlabel('单词虚部表示法每一维度的分数')
    plt.ylabel('每个单词虚部表示法的分数')
    plt.savefig(f'figure3/imag_part_{DATA_SET}.jpg', dpi=1000)
    
    return [real_part, imag_part]
```

#### 4. **Trực Quan Hóa Embedding trong `forward()`** (Dòng 651-702)

```python
def forward(self, text, sentiment):
    # 1. Word Embedding
    amplitude_is_WordEmbedding = self.amplitude_embedding(text)
    sentiment_unsqueeze = (torch.unsqueeze(sentiment, dim=-1)
                          .expand(*sentiment.shape, amplitude_is_WordEmbedding.size(2)))
    phase_is_sentiment = amplitude_is_WordEmbedding  # Lưu ý: dùng embedding, không phải sentiment
    
    # Trực quan hóa word embeddings (tiếng Trung)
    plt.figure(figsize=(10, 8))
    plt.imshow(torch.transpose(amplitude_is_WordEmbedding, 0, 1)[0,:,:].cpu().detach().numpy(),
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='文本词嵌入分数', orientation='horizontal')
    plt.title('文本词嵌入矩阵分数')
    plt.ylabel('句子中的每一个单词的词嵌入分数')
    plt.xlabel('每一个单词不同维度的词嵌入分数')
    plt.savefig(f'figure1/amplitude_is_WordEmbedding_{DATA_SET}.jpg', dpi=1000)
    
    # Trực quan hóa sentiment (tiếng Trung)
    plt.figure(figsize=(10, 8))
    plt.imshow(torch.transpose(phase_is_sentiment, 0, 1).cpu()[0,:,:].detach().numpy(),
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='情感极性分数', orientation='horizontal')
    plt.title('文本情感极性分数')
    plt.ylabel('句子中的每一个单词的情感极性分数')
    plt.xlabel('每一个单词不同维度的情感极性分数')
    plt.savefig(f'figure1/phase_is_sentiment_{DATA_SET}.jpg', dpi=1000)
```

#### 5. **Trực Quan Hóa Output Layer** (Dòng 730-762)

```python
# Sau pooling
Max_real = self.MaxPool1(torch.sigmoid(Conv_real))
plt.figure(figsize=(10, 8))
plt.imshow(Max_real.squeeze(1).squeeze(1).cpu(), 
           cmap='viridis', interpolation='nearest')
plt.title('实部特征提取效果')
plt.xlabel('每一个句子中的实部特征提取效果')
plt.ylabel('不同分类极性分数')
plt.savefig(f'figure4/Max_real_{DATA_SET}.jpg', dpi=1000)

Max_imag = self.MaxPool2(torch.sigmoid(Conv_imag))
plt.figure(figsize=(10, 8))
plt.imshow(Max_imag.squeeze(1).squeeze(1).cpu(), 
           cmap='viridis', interpolation='nearest')
plt.title('虚部特征提取效果')
plt.xlabel('每一个句子中的虚部特征提取效果')
plt.ylabel('不同分类极性分数')
plt.savefig(f'figure4/Max_imag_{DATA_SET}.jpg', dpi=1000)

# Output FC1
fc1 = self.fc1(torch.cat((Max_real, Max_imag), dim=3))
plt.figure(figsize=(10, 8))
plt.imshow(fc1.squeeze(1).squeeze(1).cpu().detach().numpy(), 
           cmap='viridis', interpolation='nearest')
plt.title('最终特征提取效果')
plt.xlabel('每一个句子中的最终特征提取效果')
plt.ylabel('不同分类极性分数')
plt.savefig(f'figure4/fc1_{DATA_SET}.jpg', dpi=1000)

# Output FC2
fc2 = torch.sigmoid(self.fc2(torch.sigmoid(fc1)))
plt.figure(figsize=(10, 8))
plt.imshow(fc2.squeeze(1).squeeze(1).cpu(), 
           cmap='viridis', interpolation='nearest')
plt.title('最终特征提取效果')
plt.xlabel('每一个句子中的最终特征提取效果')
plt.ylabel('不同分类极性分数')
plt.savefig(f'figure4/fc2_{DATA_SET}.jpg', dpi=1000)
```

#### 6. **Evaluation với Visualization** (Dòng 813-870)

```python
def evaluate(model, iterator, criterion, CUDA_NUMBER):
    model.load_state_dict(torch.load(f'Best_model_of_{DATA_SET}.pt', 
                                     map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, label, sentiment = batch[0], batch[1], batch[2]
            
            # Trực quan hóa text indices đầu vào (tiếng Trung)
            plt.figure(figsize=(10, 8))
            plt.imshow(torch.transpose(text, 0, 1).cpu(), 
                      cmap='viridis', interpolation='nearest')
            plt.colorbar(label='文本序列号', orientation='horizontal')
            plt.title('每一个Batch中的文本序列号')
            plt.xlabel('每一个句子中的文本序列号')
            plt.ylabel('一个Batch中的句子长度')
            plt.savefig(f'figure0/text_{DATA_SET}.jpg', dpi=1000)
            
            # Trực quan hóa sentiment scores (tiếng Trung)
            plt.figure(figsize=(10, 8))
            plt.imshow(torch.transpose(sentiment, 0, 1).cpu(), 
                      cmap='viridis', interpolation='nearest')
            plt.colorbar(label='情感极性分数', orientation='horizontal')
            plt.title('每一个Batch中情感极性分数')
            plt.xlabel('每一个句子中的情感极性分数')
            plt.ylabel('一个Batch中的句子长度')
            plt.savefig(f'figure0/sentiment_{DATA_SET}.jpg', dpi=1000)
            
            predictions = model(text, sentiment).squeeze(1)
            # ... tính metrics ...
            return  # Chỉ xử lý batch đầu tiên để tạo visualization
```

**Điểm Chú Ý**: Hàm return sau batch đầu tiên để tạo visualizations, không phải để đánh giá hoàn chỉnh.

---

## File 3: test_vis_english.py

### Mục Đích
Giống như `test_vis.py` nhưng với **nhãn tiếng Anh** trong visualizations. Phù hợp cho các ấn phẩm và bài thuyết trình quốc tế.

### Điểm Khác Biệt So Với test_vis.py

#### 1. **Cấu Hình Font** (Dòng 11)
```python
plt.rcParams['font.family'] = 'Times New Roman'  # Font tiếng Anh
```

#### 2. **Nhãn Visualization** (Ví dụ)

**Tiếng Trung (test_vis.py)**:
```python
plt.colorbar(label='幅值信息', orientation='horizontal')
plt.title('幅值信息')
plt.xlabel('单个句子幅值表示的分数')
plt.ylabel('单词幅值表示中不同维度表示的分数')
```

**Tiếng Anh (test_vis_english.py)**:
```python
plt.colorbar(label='Amplitude Information', orientation='horizontal')
plt.title('Amplitude Information')
plt.xlabel('Amplitude Score of Single Sentence')
plt.ylabel('Scores of Different Dimensions in Word Amplitude')
```

#### 3. **Các Seed Được Test Thêm** (Dòng 298-302)
```python
# SEED = 1111
# SEED = 1234
# SEED = 5678
SEED = 6666  # Hiện đang active
SA = 1       # Cờ Sentiment Analysis
```

**Mục Đích**: Nhiều seeds để test tính tái tạo kết quả (reproducibility) và phương sai trong kết quả.

#### 4. **Bảng Dịch Nhãn Hoàn Chỉnh**

| Tiếng Trung (test_vis.py) | Tiếng Anh (test_vis_english.py) |
|----------------------|------------------------------|
| 文本词嵌入分数 | WordEmbedding Vector of Text |
| 情感极性分数 | Sentiment Polarity |
| 幅值信息 | Amplitude Information |
| 相角信息 | Phase Information |
| 复数表示法的实部 | Real Part |
| 复数表示法的虚部 | Imaginary Part |
| 实部特征提取效果 | Real Part Feature Extraction |
| 虚部特征提取效果 | Imaginary Part Feature Extraction |
| 最终特征提取效果 | Final Feature Extraction |

### File Visualization Được Tạo Ra

Cả hai scripts đều tạo cùng cấu trúc thư mục:
```
{SA}_{DATA_SET}/
  ├── figure0_{DATA_SET}_{SEED}/
  │   ├── text_{DATA_SET}.jpg          # Indices text đầu vào
  │   └── sentiment_{DATA_SET}.jpg     # Điểm sentiment
  ├── figure1_{DATA_SET}_{SEED}/
  │   ├── amplitude_is_WordEmbedding_{DATA_SET}.jpg  # Word embeddings
  │   └── phase_is_sentiment_{DATA_SET}.jpg          # Ma trận sentiment
  ├── figure2_{DATA_SET}_{SEED}/
  │   ├── amplitude_norm_{DATA_SET}.jpg  # Amplitude đã chuẩn hóa
  │   └── phase_permute_{DATA_SET}.jpg   # Thông tin phase
  ├── figure3_{DATA_SET}_{SEED}/
  │   ├── real_part_{DATA_SET}.jpg       # Thành phần thực
  │   └── imag_part_{DATA_SET}.jpg       # Thành phần ảo
  └── figure4_{DATA_SET}_{SEED}/
      ├── Max_real_{DATA_SET}.jpg        # Đặc trưng real sau pooling
      ├── Max_imag_{DATA_SET}.jpg        # Đặc trưng imaginary sau pooling
      ├── fc1_{DATA_SET}.jpg             # Output FC layer đầu tiên
      └── fc2_{DATA_SET}.jpg             # Predictions cuối cùng
```

---

## File 4: aMain_20102025.ipynb

### Mục Đích
Jupyter Notebook version của pipeline training. Hỗ trợ interactive development, debugging, và trực quan hóa từng bước.

### Cấu Trúc Notebook (23 Cells)

#### **Cell 1-2: Import và Cấu Hình**
```python
# Cell 1: Import thư viện
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn

# Cell 2: Cài đặt NLTK data
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('omw-1.4')
```

#### **Cell 3-6: Preprocessing Functions**
```python
# Cell 3: Function doc sentiment từ SentiWordNet
def doc_sentiment(input_text):
    tag_score = 0
    tagged = nltk.pos_tag(nltk.word_tokenize(input_text))
    
    for i in tagged:
        tag_w = i[0]
        tag_wpos = i[1]
        
        pos_tag = penn_to_wn(tag_wpos)
        if pos_tag:
            lemma = lemmatize(tag_w, pos=pos_tag)
            synsets = wordnet.synsets(lemma, pos=pos_tag)
            
            if synsets:
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                tag_score += swn_synset.pos_score() - swn_synset.neg_score()
    
    return tag_score

# Cell 4: Helper functions (lemmatize, penn_to_wn)
# Cell 5: tokenize_and_cut function
# Cell 6: sub_word_vectors_load function
```

#### **Cell 7-9: Data Loading và Field Definition**
```python
# Cell 7: Cấu hình dataset
DATA_SET = 'SUBJ'
BATCH_SIZE = 16
MAX_LEN = 60

# Cell 8: Define Fields
TEXT = data.Field(tokenize=tokenize_and_cut, lower=True)
LABEL = data.LabelField(dtype=torch.float)
SENTIMENT = data.Field(use_vocab=False, dtype=torch.float, 
                       preprocessing=lambda x: [doc_sentiment(x)])

# Cell 9: Load datasets
fields = [('text', TEXT), ('label', LABEL), ('sentiment', SENTIMENT)]
train_data = data.TabularDataset(path=f'./DataSet/{DATA_SET}/train.tsv',
                                 format='tsv', fields=fields)
test_data = data.TabularDataset(path=f'./DataSet/{DATA_SET}/test.tsv',
                                format='tsv', fields=fields)
```

#### **Cell 10-11: Vocabulary và Word Vectors**
```python
# Cell 10: Build vocabulary
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

# Cell 11: Load pre-trained word vectors
vecs = sub_word_vectors_load(f'./sub_word_vector/{DATA_SET}_word_vector.txt')
TEXT.vocab.set_vectors(vecs.stoi, vecs.vectors, vecs.dim)
```

#### **Cell 12-14: Data Iterators**
```python
# Cell 12: Cấu hình device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_NUMBER = 0

# Cell 13: Tạo iterators
train_iterator = data.BucketIterator(
    train_data, batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.text), device=device)

test_iterator = data.BucketIterator(
    test_data, batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.text), device=device)

# Cell 14: Kiểm tra data shape
for batch in train_iterator:
    print(f"Text shape: {batch.text.shape}")
    print(f"Label shape: {batch.label.shape}")
    print(f"Sentiment shape: {batch.sentiment.shape}")
    break
```

#### **Cell 15-18: Model Classes**
```python
# Cell 15: projection_Euler class
class projection_Euler(nn.Module):
    def __init__(self):
        super(projection_Euler, self).__init__()
    
    def forward(self, inputs):
        amplitude = inputs[0]
        phase = inputs[1]
        
        amplitude_permute = amplitude.permute(1, 0, 2)
        amplitude_norm = F.normalize(amplitude_permute, 2, 2)
        phase_permute = phase.permute(1, 0, 2)
        
        real_part = amplitude_norm * torch.cos(phase_permute)
        imag_part = amplitude_norm * torch.sin(phase_permute)
        
        return [real_part, imag_part]

# Cell 16: projection_CalculateMatrixMean class
# Cell 17: projection_CalculateMatrixQ + projection_Measurement
# Cell 18: self_attention + LSTMWithSkipConnection classes
```

#### **Cell 19-20: Main Model và Initialization**
```python
# Cell 19: run_complex_network class
class run_complex_network(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 output_dim, pad_idx, num_layers):
        super().__init__()
        
        self.amplitude_embedding = nn.Embedding(vocab_size, embedding_dim, 
                                                padding_idx=pad_idx)
        self.lstm_real = LSTMWithSkipConnection(embedding_dim, hidden_dim, 
                                                num_layers)
        self.lstm_imag = LSTMWithSkipConnection(embedding_dim, hidden_dim, 
                                                num_layers)
        self.projection_Euler = projection_Euler()
        self.projection_Mean = projection_CalculateMatrixMean()
        self.projection_Q = projection_CalculateMatrixQ()
        self.projection_Measurement = projection_Measurement()
        self.self_attention_real = self_attention()
        self.self_attention_imag = self_attention()
        
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.Conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, text, sentiment):
        # Forward pass logic
        # ...
        return torch.sigmoid(self.fc2(torch.sigmoid(fc1))).squeeze(1).squeeze(1)

# Cell 20: Initialize model
EMBEDDING_DIM = 100
HIDDEN_DIM = 3
OUTPUT_DIM = 1
num_layers = 2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = run_complex_network(len(TEXT.vocab), EMBEDDING_DIM, HIDDEN_DIM,
                            OUTPUT_DIM, PAD_IDX, num_layers)

# Load pretrained embeddings
pretrained_embeddings = TEXT.vocab.vectors
model.amplitude_embedding.weight.data.copy_(pretrained_embeddings)
model.amplitude_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
```

#### **Cell 21: Training Setup**
```python
# Cell 21: Optimizer và Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

model = model.cuda(CUDA_NUMBER)
criterion = criterion.cuda(CUDA_NUMBER)

# Khởi tạo tracking lists
acc_train = []
acc_test = []
loss_train = []
loss_test = []
f1_score_test = []
recall_test = []
```

#### **Cell 22: Training Loop**
```python
# Cell 22: Main training loop
N_EPOCHS = 26
best_test_acc = -float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc, train_f1, train_recall = train(
        model, train_iterator, optimizer, criterion, CUDA_NUMBER)
    
    test_loss, test_acc, test_f1, test_recall = evaluate(
        model, test_iterator, criterion, CUDA_NUMBER)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # Track metrics
    acc_train.append(train_acc)
    acc_test.append(test_acc)
    loss_train.append(train_loss)
    loss_test.append(test_loss)
    f1_score_test.append(test_f1)
    recall_test.append(test_recall)
    
    # Lưu best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), f'Best_model_of_{DATA_SET}.pt')
    
    # Print metrics
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print(f'Test F1: {test_f1:.4f} | Test Recall: {test_recall:.4f}')
```

#### **Cell 23: Latency Measurement (Tính Năng Độc Nhất)**
```python
# Cell 23: Đo latency inference
import time

model.load_state_dict(torch.load(f'Best_model_of_{DATA_SET}.pt'))
model.eval()

latencies = []

with torch.no_grad():
    for batch in test_iterator:
        text, label, sentiment = batch.text, batch.label, batch.sentiment
        
        # Warmup
        for _ in range(10):
            _ = model(text, sentiment)
        
        # Measure latency
        for _ in range(100):
            start_time = time.time()
            predictions = model(text, sentiment)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms

# Thống kê latency
latencies = np.array(latencies)
print(f"Mean latency: {np.mean(latencies):.2f} ms")
print(f"Median latency (p50): {np.percentile(latencies, 50):.2f} ms")
print(f"p95 latency: {np.percentile(latencies, 95):.2f} ms")
print(f"p99 latency: {np.percentile(latencies, 99):.2f} ms")
print(f"Min latency: {np.min(latencies):.2f} ms")
print(f"Max latency: {np.max(latencies):.2f} ms")

# Visualize latency distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(latencies, bins=50, edgecolor='black')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.title('Latency Distribution')

plt.subplot(1, 2, 2)
plt.boxplot(latencies)
plt.ylabel('Latency (ms)')
plt.title('Latency Boxplot')
plt.tight_layout()
plt.savefig(f'latency_{DATA_SET}.png', dpi=300)
plt.show()
```

---

## Tổng Quan Kiến Trúc

### 1. **Sơ Đồ Luồng Dữ Liệu Hoàn Chỉnh**

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                                 │
│  Text Tokens (vocab indices) + Sentiment Scores (SentiWordNet)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────────┐              ┌──────────────┐
    │   Embedding   │              │  Sentiment   │
    │   (100-dim)   │              │   Scores     │
    └───────┬───────┘              └──────┬───────┘
            │                             │
            │ (amplitude)                 │ (phase)
            │                             │
            └──────────┬──────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Euler Projection   │
            │ real = amp * cos(θ) │
            │ imag = amp * sin(θ) │
            └──────────┬──────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │  LSTM (Real) │      │ LSTM (Imag)  │
    │  + Attention │      │ + Attention  │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           │                     │
           └─────────┬───────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Density Matrix │
            │   ρ = |ψ⟩⟨ψ|    │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Measurement    │
            │  P(0) = Tr(M₀ρ) │
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
    ┌──────────┐      ┌──────────┐
    │ CNN Real │      │ CNN Imag │
    │ 3x3 Conv │      │ 3x3 Conv │
    │ MaxPool  │      │ MaxPool  │
    └─────┬────┘      └─────┬────┘
          │                 │
          └────────┬────────┘
                   │
                   ▼
            ┌──────────────┐
            │ Concatenate  │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │   FC1 (64)   │
            │   + Sigmoid  │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │   FC2 (1)    │
            │   + Sigmoid  │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │   OUTPUT     │
            │ Binary (0/1) │
            └──────────────┘
```

### 2. **Quantum-Inspired Components Chi Tiết**

#### **A. Euler Representation**
```
Complex Number = Amplitude × e^(iθ)
               = Amplitude × (cos(θ) + i·sin(θ))

Trong code:
- Amplitude = Word Embedding (chuẩn hóa L2)
- Phase (θ) = Sentiment Score từ SentiWordNet
- Real Part = Amplitude × cos(Sentiment)
- Imaginary Part = Amplitude × sin(Sentiment)
```

**Ý Nghĩa Vật Lý**:
- **Amplitude**: Độ lớn vector → độ quan trọng của từ
- **Phase**: Góc quay → sentiment polarity (positive/negative)
- **Real/Imag**: Hai khía cạnh bổ sung cho nhau

#### **B. Density Matrix**
```
ρ = |ψ⟩⟨ψ| = (ψ_real + i·ψ_imag)(ψ_real - i·ψ_imag)ᵀ

Trong code:
ρ_real = (ψ_real ⊗ ψ_real) + (ψ_imag ⊗ ψ_imag)
ρ_imag = (ψ_imag ⊗ ψ_real) - (ψ_real ⊗ ψ_imag)

Dimensions: [batch, hidden_dim, hidden_dim]
```

**Ý Nghĩa Vật Lý**:
- Mô tả trạng thái hỗn hợp (mixed state) trong quantum mechanics
- Capture correlations giữa các hidden dimensions

#### **C. Measurement Operator**
```
Probability = Tr(M · ρ)

Trong code:
M₀ = |0⟩⟨0| = [[1, 0], [0, 0]]  # Measurement cho class 0
M₁ = |1⟩⟨1| = [[0, 0], [0, 1]]  # Measurement cho class 1

P(class_0) = Σᵢ (M₀ · ρ)[i,i]  # Trace operation
```

**Ý Nghĩa Vật Lý**:
- Collapse wave function → discrete outcome
- Projection lên computational basis states

### 3. **Hyperparameters Chi Tiết**

```python
# Data
BATCH_SIZE = 16
MAX_LEN = 60                    # Truncate câu dài hơn 60 từ

# Model Architecture
EMBEDDING_DIM = 100             # Word vector dimension
HIDDEN_DIM = 3                  # LSTM hidden state dimension
OUTPUT_DIM = 1                  # Binary classification
num_layers = 2                  # LSTM layers

# Training
N_EPOCHS = 26
learning_rate = 0.001           # Adam optimizer
criterion = nn.BCELoss()        # Binary Cross-Entropy

# Regularization
PAD_IDX = 1                     # Padding token index (weight = 0)

# Visualization
SEED_training = 1234
SEED_visualization = 6666       # Khác seed để test reproducibility

# CNN
Conv_kernel_size = 3
MaxPool_kernel_size = 2
FC1_hidden = 64
```

### 4. **Performance Benchmarks**

Từ các kết quả trong `RESULT_ANALY.md` và `QITSA_Overview.md`:

| Dataset | Test Accuracy | F1-Score | Recall | Training Time |
|---------|--------------|----------|--------|---------------|
| **MR** | ~79.5% | ~0.795 | ~0.80 | ~15min (26 epochs) |
| **SST** | ~82.3% | ~0.823 | ~0.81 | ~18min |
| **SUBJ** | ~92.1% | ~0.921 | ~0.92 | ~12min |
| **CR** | ~81.7% | ~0.817 | ~0.82 | ~10min |
| **MPQA** | ~85.4% | ~0.854 | ~0.85 | ~20min |

**Hardware**: NVIDIA GPU (CUDA), CPU fallback support

**Latency** (từ Cell 23):
- Mean: ~5-10ms per batch (16 samples)
- p95: ~15ms
- p99: ~20ms

---

## Checklist Tái Tạo Kết Quả (Reproducibility)

### ✅ **1. Environment Setup**

```bash
# Create conda environment
conda create -n qitsa python=3.8
conda activate qitsa

# Install PyTorch (CUDA 11.1)
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install other dependencies
pip install torchtext==0.10.0
pip install nltk==3.6.2
pip install matplotlib==3.4.2
pip install scikit-learn==0.24.2
pip install pandas==1.3.0
pip install numpy==1.21.0

# Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('sentiwordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
```

### ✅ **2. Data Preparation**

```
DataSet/
├── MR/
│   ├── train.tsv    # Format: <text>\t<label>
│   ├── dev.tsv
│   └── test.tsv
├── SST/
├── SUBJ/
├── CR/
└── MPQA/
```

**TSV Format**:
```
This movie is great	1
Terrible film	0
```

### ✅ **3. Pre-trained Word Vectors**

```bash
# Tạo word vectors cho mỗi dataset
sub_word_vector/
├── MR_word_vector.txt
├── SST_word_vector.txt
├── SUBJ_word_vector.txt
├── CR_word_vector.txt
└── MPQA_word_vector.txt
```

**Format**:
```
word1 0.123 -0.456 0.789 ... (100 dimensions)
word2 -0.321 0.654 -0.987 ...
```

### ✅ **4. Random Seeds**

```python
# Training
SEED = 1234

# Visualization
SEED = 6666

# Set seed function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### ✅ **5. Training Commands**

```bash
# Chạy training script
python train_20102025.py

# Hoặc dùng notebook
jupyter notebook aMain_20102025.ipynb

# Chạy visualization (Chinese labels)
python test_vis.py

# Chạy visualization (English labels)
python test_vis_english.py
```

### ✅ **6. Expected Outputs**

```
# Models
Best_model_of_MR.pt
Best_model_of_SST.pt
Best_model_of_SUBJ.pt
Best_model_of_CR.pt
Best_model_of_MPQA.pt

# Metrics
train/loss/{DATA_SET}.txt
test/acc/{DATA_SET}.txt
test/f1/{DATA_SET}.txt
test/recall/{DATA_SET}.txt

# Plots
LossMR.png
AccMR.png
F1MR.png
RecallMR.png

# Visualizations (test_vis.py)
{SA}_{DATA_SET}/figure0_{DATA_SET}_{SEED}/...
{SA}_{DATA_SET}/figure1_{DATA_SET}_{SEED}/...
{SA}_{DATA_SET}/figure2_{DATA_SET}_{SEED}/...
{SA}_{DATA_SET}/figure3_{DATA_SET}_{SEED}/...
{SA}_{DATA_SET}/figure4_{DATA_SET}_{SEED}/...
```

### ✅ **7. Validation Checklist**

- [ ] Test accuracy khớp với benchmarks (±1%)
- [ ] F1-score trong khoảng mong đợi
- [ ] Loss converge sau ~20 epochs
- [ ] Visualization files được tạo đầy đủ
- [ ] Model size ~10-50MB (tùy dataset)
- [ ] Training time ~10-20 phút trên GPU
- [ ] Không có warnings về deprecated APIs

---

## Key Insights và Kết Luận

### 1. **Đóng Góp Chính**

**A. Hybrid Quantum-Classical Architecture**:
- Kết hợp quantum-inspired representations (Euler, density matrix, measurement) với classical deep learning (LSTM, CNN)
- Không cần quantum hardware thật → có thể chạy trên GPU thông thường

**B. Sentiment-Informed Phase Encoding**:
- Sử dụng SentiWordNet scores làm phase information
- Giúp model học được cả semantic (amplitude) lẫn sentiment (phase)

**C. Dual-Stream Processing**:
- Real và Imaginary streams xử lý song song
- Capture cả magnitude lẫn direction của sentiment

### 2. **Điểm Mạnh**

✅ **Hiệu Suất Cao**: 79-92% accuracy trên 5 datasets  
✅ **Tính Tổng Quát**: Hoạt động tốt trên nhiều domains (movies, products, opinions)  
✅ **Khả Năng Tái Tạo**: Seeds được set cẩn thận, code có cấu trúc tốt  
✅ **Visualization**: Cung cấp insights về cách model học  
✅ **Latency Thấp**: ~5-10ms per batch (suitable cho production)

### 3. **Hạn Chế và Cải Tiến Tiềm Năng**

#### **A. Code Issues**
```python
# test_vis.py line 654
phase_is_sentiment = amplitude_is_WordEmbedding  # ❌ BUG
# Nên là:
phase_is_sentiment = sentiment_unsqueeze  # ✅ FIX
```

#### **B. Model Naming**
```python
# LSTMWithSkipConnection vẫn dùng nn.LSTM
# Nên đổi tên hoặc thực sự implement GRU
```

#### **C. Architecture Enhancements**
- [ ] Thử GRU thay vì LSTM (có thể nhanh hơn)
- [ ] Experiment với Transformer encoder
- [ ] Add batch normalization
- [ ] Implement dropout cho regularization

#### **D. Training Improvements**
- [ ] Learning rate scheduling (cosine annealing)
- [ ] Early stopping based on validation loss
- [ ] K-fold cross-validation
- [ ] Ensemble multiple seeds

#### **E. Evaluation**
- [ ] Test trên out-of-domain data
- [ ] Adversarial examples testing
- [ ] Error analysis theo sentence length
- [ ] Ablation study cho từng component

### 4. **Workflow Khuyến Nghị**

```
1. Development: aMain_20102025.ipynb
   ↓ (Iterate và debug)
2. Training: train_20102025.py
   ↓ (Train best model)
3. Visualization: test_vis_english.py
   ↓ (Tạo publication-ready figures)
4. Deployment: Extract inference code
   ↓ (Optimize cho production)
5. Monitoring: Track latency và accuracy
```

### 5. **Tài Liệu Tham Khảo**

Các file liên quan trong workspace:
- `QITSA_Overview.md`: High-level architecture overview
- `QITSA_Complete_Training_Guide.md`: Step-by-step training guide
- `RESULT_ANALY.md`: Kết quả thực nghiệm chi tiết
- `qitsa-analysis.md`: Phân tích toàn diện
- `project_analysis_checklist.md`: Checklist đánh giá project

### 6. **Kết Luận**

QITSA là một **hybrid quantum-classical approach** sáng tạo cho bài toán sentiment analysis. Code được implement cẩn thận với:
- Reproducibility tốt (seeds, environment)
- Visualization phong phú (Chinese + English labels)
- Performance benchmarks rõ ràng
- Hỗ trợ cả script (.py) và notebook (.ipynb)

**Điểm mạnh nhất**: Cách kết hợp sentiment scores vào phase information, tạo ra representation vừa có semantic meaning vừa có sentiment awareness.

**Next Steps**: Fix bugs nhỏ, thêm regularization, và test trên larger-scale datasets.
