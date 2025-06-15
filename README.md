## Tổng quan

Đây là project huấn luyện và kiểm thử AI chơi game Pokémon Red trên giả lập Game Boy, sử dụng các thuật toán PPO của reinforcement learning. Project hỗ trợ cả việc train model mới và test các model đã huấn luyện.

---

## Hướng dẫn cài đặt trên Windows

### 1. Cài đặt Python

- Tải Python 3.11.6 tại: https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe  
- Khi cài đặt, nhớ tick **"Add Python to PATH"**.

### 2. Cài đặt Git

- Tải Git tại: https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe  
- Cài đặt theo các bước minh họa:
    - ![Bước 1](https://i.postimg.cc/DyJ1m4YG/image.png)
    - ![Bước 2](https://i.postimg.cc/C5WBbJ6q/image.png)
    - ![Bước 3](https://i.postimg.cc/hPd7KWfb/image.png)
    - ![Bước 4](https://i.postimg.cc/LsNJxmbR/image.png)
    - ![Bước 5](https://i.postimg.cc/1XN4Rw55/image.png)
    - ![Bước 6](https://i.postimg.cc/HjDnrnV0/image.png)
    - ![Bước 7](https://i.postimg.cc/W17mPLHY/image.png)
    - ![Bước 8](https://i.postimg.cc/cJ9Qr7ZQ/image.png)
    - ![Bước 9](https://i.postimg.cc/638CR75F/image.png)
    - ![Bước 10](https://i.postimg.cc/2jB4SBtZ/image.png)
    - ![Bước 11](https://i.postimg.cc/sD4GMP4v/image.png)
    - ![Bước 12](https://i.postimg.cc/bvWs7mpF/image.png)

### 3. Cài đặt Microsoft C++ Build Tools

- Tải tại: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Cài đặt như hình minh họa:
    - ![Build Tools](https://i.postimg.cc/Yq15fqds/image.png)
    - ![Windows 11](https://i.postimg.cc/VkKHDSnD/image.png)
    - ![Windows 10](https://i.postimg.cc/43KwPWJx/image.png)

---

## Cài đặt Python package

1. Mở **cmd**.
2. Di chuyển vào thư mục `baselines` của project:
    ```sh
    cd "C:\Users\YourWindowsUsername\Downloads\PokemonRedExperiments\baselines"
    ```
3. Cài đặt các thư viện cần thiết:
    ```sh
    pip install -r requirements.txt
    ```
    Nếu có nhiều phiên bản Python, dùng đường dẫn tuyệt đối:
    ```sh
    "%localappdata%\Programs\Python\Python311\Scripts\pip.exe" install -r requirements.txt
    ```

---

## Cách chạy project

### **A. Test model đã huấn luyện**

Chạy file [run_pretrained_interactive.py](http://_vscodecontentref_/0) để test model:
```sh
python run_pretrained_interactive.py
```
### **B. Train model mới (lưu ý, cần rất nhiêu core cpu)**

Chạy file run_baseline_parallel_fast.py để bắt đầu train:
```sh
python run_baseline_parallel.py
```
hoặc
```sh
python run_baseline_parallel_fast.py
```