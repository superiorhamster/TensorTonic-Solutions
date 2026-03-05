import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    
    # Bước 1: Giới hạn giá trị dự đoán để tránh lỗi log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # Bước 2: Lấy ra các xác suất tương ứng với nhãn đúng
    # (Dùng fancy indexing của NumPy)
    samples = len(y_true)
    correct_confidences = y_pred[range(samples), y_true]
    
    # Bước 3: Tính toán giá trị logarit âm và trung bình cộng
    negative_log_likelihoods = -np.log(correct_confidences)
    loss = np.mean(negative_log_likelihoods)
    
    return loss