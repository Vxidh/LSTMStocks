import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel

def prepare_data(data, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i])
        y.append(data_scaled[i, data.columns.get_loc('Close')])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_model(X_train, y_train, X_test, y_test, input_size, hidden_size, num_layers, learning_rate, epochs, dropout):
    model = LSTMModel(input_size=input_size, hidden_layer_size=hidden_size, num_layers=num_layers, dropout=dropout)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    train_losses = []
    val_losses = []
    batch_size = 32
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        model.eval()
        with torch.no_grad():
            val_output = model(X_test)
            val_loss = criterion(val_output, y_test)
            val_losses.append(val_loss.item())
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    return model, train_losses, val_losses
