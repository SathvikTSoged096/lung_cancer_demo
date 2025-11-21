import tensorflow as tf
import numpy as np
import cv2

# Load your saved model
def load_model(model_path="resnet50_lung_cancer.h5"):
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess uploaded image
def preprocess_image(img_path, target_size=(224,224)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    orig_img = img.copy()
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img, orig_img

# Generate Grad-CAM heatmap and overlay
def generate_gradcam(model, img_array, orig_img, last_conv_layer_name="conv5_block3_out", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    
    # Find bounding box of high-intensity regions
    _, thresh = cv2.threshold(heatmap[:,:,2], 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay_img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    return overlay_img, predictions.numpy()[0]
