import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import tensorflow as tf

# Gesture definitions (must match model training labels)
gestures = [
    {'emoji': '🖐', 'name': 'open_hand'},
    {'emoji': '👌', 'name': 'ok'},
    {'emoji': '🤏', 'name': 'pinch'},
    {'emoji': '✌️', 'name': 'victory'},
    {'emoji': '🤟', 'name': 'love_you'},
    {'emoji': '👍', 'name': 'thumbs_up'},
    {'emoji': '👎', 'name': 'thumbs_down'},
    {'emoji': '👈', 'name': 'left_point'},
    {'emoji': '✊', 'name': 'fist'},
    {'emoji': '🫶', 'name': 'heart_hands'},
    {'emoji': '🙏', 'name': 'pray'},
    {'emoji': '🤞', 'name': 'fingers_crossed'},
    {'emoji': '💪', 'name': 'flex'},
    {'emoji': '🤌', 'name': 'pinched_fingers'}
]

def model_svm(train_path, test_path, model_save_path, num_classes, params):
    """
    Trains an SVM model using separate train and test CSV files and saves results.
    Args:
        train_path (str): Path to the training CSV dataset (label in first column).
        test_path (str): Path to the test CSV dataset (label in first column).
        model_save_path (str): Base filename for saving the trained model.
        num_classes (int): Number of classes for classification.
        params (dict): Hyperparameters: C, kernel, gamma, random_seed.
    Returns:
        dict: test_accuracy, result_folder, history (train/test accuracy).
    """
    # Hyperparameters
    svm_C       = params.get("C", 1.0)
    svm_kernel  = params.get("kernel", "rbf")
    svm_gamma   = params.get("gamma", "scale")
    random_seed = params.get("random_seed", 42)
    np.random.seed(random_seed)

    # Load train and test data
    train = np.loadtxt(train_path, delimiter=',', dtype='float32')
    test = np.loadtxt(test_path, delimiter=',', dtype='float32')
    y_train = train[:, 0].astype(np.int32)
    X_train = train[:, 1:]
    y_test  = test[:, 0].astype(np.int32)
    X_test  = test[:, 1:]

    # Build and fit SVM
    model = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma,
                random_state=random_seed, probability=True)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    print(f"Training accuracy: {train_acc}, Test accuracy: {test_acc}")

    # Results folder
    folder = f"results_SVM_C{svm_C}_kernel{svm_kernel}_gamma{svm_gamma}"
    os.makedirs(folder, exist_ok=True)

    # map numeric labels to gesture names
    label_indices = sorted(set(y_test))
    label_names   = [gestures[i]['name'] for i in label_indices]
    cm = confusion_matrix(y_test, model.predict(X_test), labels=label_indices)

    # make figure larger so names don’t overlap
    plt.figure(figsize=(12,10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(folder, "confusion_matrix.png"))
    plt.close()

    # Classification report
    report = classification_report(y_test, model.predict(X_test))
    with open(os.path.join(folder, "classification_report.txt"), 'w') as f:
        f.write(report)

    # Save model
    save_path = os.path.join(folder, os.path.basename(model_save_path))
    joblib.dump(model, save_path)

    return {"test_accuracy": test_acc, "result_folder": folder,
            "history": {"train_accuracy": train_acc, "test_accuracy": test_acc}}


def model_rf(train_path, test_path, model_save_path, num_classes, params):
    """
    Trains a Random Forest model using separate train and test CSV files and saves results.
    Args:
        train_path, test_path: CSV paths with label in first column.
        model_save_path: Base filename for saving the model.
        num_classes: Number of classes.
        params: n_estimators, max_depth, random_seed.
    Returns:
        dict: test_accuracy, result_folder, history.
    """
    # Hyperparameters
    n_estimators = params.get("n_estimators", 100)
    max_depth    = params.get("max_depth", None)
    random_seed  = params.get("random_seed", 42)
    np.random.seed(random_seed)

    # Load
    train = np.loadtxt(train_path, delimiter=',', dtype='float32')
    test = np.loadtxt(test_path, delimiter=',', dtype='float32')
    y_train = train[:, 0].astype(np.int32)
    X_train = train[:, 1:]
    y_test  = test[:, 0].astype(np.int32)
    X_test  = test[:, 1:]

    # Model
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   random_state=random_seed)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    print(f"Training accuracy: {train_acc}, Test accuracy: {test_acc}")

    # Folder
    folder = f"results_RF_estimators{n_estimators}_depth{max_depth or 'None'}"
    os.makedirs(folder, exist_ok=True)

    # map numeric labels to gesture names
    label_indices = sorted(set(y_test))
    label_names   = [gestures[i]['name'] for i in label_indices]
    cm = confusion_matrix(y_test, model.predict(X_test), labels=label_indices)

    # make figure larger so names don’t overlap
    plt.figure(figsize=(12,10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(folder, "confusion_matrix.png"))
    plt.close()

    report = classification_report(y_test, model.predict(X_test))
    with open(os.path.join(folder, "classification_report.txt"), 'w') as f:
        f.write(report)

    save_path = os.path.join(folder, os.path.basename(model_save_path))
    joblib.dump(model, save_path)

    return {"test_accuracy": test_acc, "result_folder": folder,
            "history": {"train_accuracy": train_acc, "test_accuracy": test_acc}}


def model_nn(train_path, test_path, model_save_path, num_classes, params):
    """
    Trains a neural network model using separate train/test CSVs and saves results.
    Args:
        train_path, test_path: CSV files with label in first column.
        model_save_path: Filename for final model.
        num_classes: Number of classes.
        params: dropout1, dropout2, dense_units1, dense_units2, epochs, batch_size, random_seed.
    Returns:
        dict: val_loss, val_accuracy, result_folder, history.
    """
    # Hyperparams
    dropout1     = params.get("dropout1", 0.2)
    dropout2     = params.get("dropout2", 0.4)
    dense1       = params.get("dense_units1", 20)
    dense2       = params.get("dense_units2", 10)
    epochs       = params.get("epochs", 1000)
    batch_size   = params.get("batch_size", 128)
    random_seed  = params.get("random_seed", 42)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Load
    train = np.loadtxt(train_path, delimiter=',', dtype='float32')
    test  = np.loadtxt(test_path, delimiter=',', dtype='float32')
    y_train = train[:, 0].astype(np.int32)
    X_train = train[:, 1:]
    y_test  = test[:, 0].astype(np.int32)
    X_test  = test[:, 1:]

    # Input dim
    input_dim = X_train.shape[1]

    # Build
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dropout(dropout1),
        tf.keras.layers.Dense(dense1, activation='relu'),
        tf.keras.layers.Dropout(dropout2),
        tf.keras.layers.Dense(dense2, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Folder
    folder = f"results_dense{dense1}_{dense2}_dropout{dropout1}_{dropout2}_batch{batch_size}_epochs{epochs}"
    os.makedirs(folder, exist_ok=True)

    # Callbacks
    checkpoint = os.path.join(folder, os.path.basename(model_save_path))
    cp_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint, save_weights_only=False, verbose=1)
    es_cb = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    # Train
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[cp_cb, es_cb])

    # Save loss/accuracy plots
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend();
    plt.savefig(os.path.join(folder, 'loss_graph.png')); plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend();
    plt.savefig(os.path.join(folder, 'accuracy_graph.png')); plt.close()

    # Evaluate
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

    # Confusion matrix & report
    preds = np.argmax(model.predict(X_test), axis=1)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(os.path.join(folder, 'confusion_matrix.png')); plt.close()

    report = classification_report(y_test, preds)
    with open(os.path.join(folder, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Final model already saved via checkpoint
    return {"val_loss": val_loss, "val_accuracy": val_acc,
            "result_folder": folder,
            "history": history.history}
