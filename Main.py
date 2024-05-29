# Image Processing & Pattern Recognition Project.
# Edge Detection (Roberts, Sobel, and Prewitt operators).
# Made by Laith Ghnemat 1200610.

import cv2      # OpenCV library, used for computer vision tasks.
import numpy as np      # numpy library, used for numerical operations.
import matplotlib.pyplot as plt     # This library is used for plotting images & graphs.

image_path = "lena.jpg"     # Get the path for Lena's image file.
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    # Read the image from the path in grayscale mode.

#############################################################################################################
# ############################################ Robert Operator ##############################################


def roberts_operator(img):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=int)    # Roberts cross kernel for x direction.
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)    # Roberts cross kernel for y direction.

    # We used 16 bits signed integer to handle negative values in the convolution result.
    x = cv2.filter2D(img, cv2.CV_16S, kernel_x)     # Apply convolution operation for x kernel.
    y = cv2.filter2D(img, cv2.CV_16S, kernel_y)     # Apply convolution operation for y kernel.

    abs_x = cv2.convertScaleAbs(x)      # Converts x result to an absolute value & scales it to 8 bits.
    abs_y = cv2.convertScaleAbs(y)      # Converts y result to an absolute value & scales it to 8 bits.

    # Combines x and y gradients with equal weights(0.5).
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
#############################################################################################################
# ############################################# Sobel Operator ##############################################


def sobel_operator(img):
    # We have taken the kernel size to be 3X3.
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)  # Derivative in the x direction.
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)  # Derivative in the y direction.

    abs_grad_x = cv2.convertScaleAbs(grad_x)  # Converts x result to an absolute value & scales it to 8 bits.
    abs_grad_y = cv2.convertScaleAbs(grad_y)  # Converts y result to an absolute value & scales it to 8 bits.

    # Combines x and y gradients with equal weights(0.5).
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#############################################################################################################
# ############################################# Prewitt Operator ############################################


def prewitt_operator(img):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)    # Prewitt kernel for x direction.
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)    # Prewitt kernel for y direction.

    x = cv2.filter2D(img, cv2.CV_16S, kernel_x)     # Apply convolution operation for x kernel.
    y = cv2.filter2D(img, cv2.CV_16S, kernel_y)     # Apply convolution operation for y kernel.

    abs_x = cv2.convertScaleAbs(x)      # Converts x result to an absolute value & scales it to 8 bits.
    abs_y = cv2.convertScaleAbs(y)      # Converts y result to an absolute value & scales it to 8 bits.

    # Combines x and y gradients with equal weights(0.5).
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
#############################################################################################################
# ##################################### Applying Edge Detection Operators ###################################


roberts_edges = roberts_operator(image)
sobel_edges = sobel_operator(image)
prewitt_edges = prewitt_operator(image)

manual_edges = np.zeros_like(image)     # Array of zeros with the same shape as the original image.
manual_edges[50:150, 50:150] = 255  # We take a square region in the image.
#############################################################################################################
# ############################################## Comparing Edges ############################################


def compare_edges(detected, manual):
    # Create a blank color image.
    comparison = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    comparison[..., 0] = detected  # Implement detected edges in red color.
    comparison[..., 1] = manual  # Implement manual edges in green color.

    return comparison
#############################################################################################################
# ######################################## Generate Comparisons Edges #######################################


# Create comparison images by applying compare_edges function for each edge detection.
comparison_roberts = compare_edges(roberts_edges, manual_edges)
comparison_sobel = compare_edges(sobel_edges, manual_edges)
comparison_prewitt = compare_edges(prewitt_edges, manual_edges)
#############################################################################################################
# ############################################## Display Results ############################################


fig, axs = plt.subplots(3, 2, figsize=(15, 15))     # Create a 3X2 grid of subplots.

# Display the images and comparisons in the subplots.
axs[0, 0].imshow(roberts_edges, cmap='gray')
axs[0, 1].imshow(comparison_roberts)
axs[1, 0].imshow(sobel_edges, cmap='gray')
axs[1, 1].imshow(comparison_sobel)
axs[2, 0].imshow(prewitt_edges, cmap='gray')
axs[2, 1].imshow(comparison_prewitt)

# Set the title for each subplot.
axs[0, 0].set_title('Roberts Edge Detection')
axs[0, 1].set_title('Roberts Comparison')
axs[1, 0].set_title('Sobel Edge Detection')
axs[1, 1].set_title('Sobel Comparison')
axs[2, 0].set_title('Prewitt Edge Detection')
axs[2, 1].set_title('Prewitt Comparison')

for ax in axs.flat:
    ax.axis('off')      # Turn off the axis for each subplot.

plt.show()      # Display the plot.
#############################################################################################################
# ############################################ Results Evaluation ###########################################


def evaluate_edges(detected, manual):   # This function will compute precision and recall.
    matched = np.sum((detected > 0) & (manual > 0))     # Total number of pixels(detected & manual edges).
    print('Matched: ', matched)

    total_manual = np.sum(manual > 0)       # Total number of manual edge pixels.
    print('Total Manual: ', total_manual)

    total_detected = np.sum(detected > 0)       # Total number of detected edge pixels.
    print('Total Detected: ', total_detected)

    # Precision = ratio of total matched edges to total detected edges.
    precision_value = matched / total_detected if total_detected else 0

    # Recall = ratio of total matched edges to total manual edges.
    recall_value = matched / total_manual if total_manual else 0

    return precision_value, recall_value
#############################################################################################################
# ######################################## Matched, Manual & Detected #######################################


print('############################################## Robert ##############################################')
precision_roberts, recall_roberts = evaluate_edges(roberts_edges, manual_edges)
print(f"Precision: {precision_roberts}")
print(f"Recall: {recall_roberts}")
print()
print('############################################## Sobel ###############################################')
precision_sobel, recall_sobel = evaluate_edges(sobel_edges, manual_edges)
print(f"Precision: {precision_sobel}")
print(f"Recall: {recall_sobel}")
print()
print('############################################## Prewitt #############################################')
precision_prewitt, recall_prewitt = evaluate_edges(prewitt_edges, manual_edges)
print(f"Precision: {precision_prewitt}")
print(f"Recall: {recall_prewitt}")
print()
#############################################################################################################
# ###############################################  Thresholding #############################################


# Function to apply an edge detection operator, threshold the result & evaluate it.
def adjust_threshold(img, operator, manual, threshold_value):
    detected_edges = operator(img)
    # Applies a binary threshold to the detected edges.
    _, binary_edges = cv2.threshold(detected_edges, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_edges, evaluate_edges(binary_edges, manual)


threshold_values = range(50, 200, 50)

print('############################################# Robert ###############################################')
fig, axs = plt.subplots(1, len(threshold_values), figsize=(20, 5))
for i, threshold in enumerate(threshold_values):
    print('################################'f' Threshold: {threshold} ' '##################################')
    # Computes precision and recall for the given threshold.
    binary_edges, (precision, recall) = adjust_threshold(image, roberts_operator, manual_edges, threshold)
    print(f"Precision: {precision}, Recall: {recall}")
    print()

    axs[i].imshow(binary_edges, cmap='gray')
    axs[i].set_title(f'Threshold: {threshold}\nPrecision: {precision:.2f}, Recall: {recall:.2f}')
    axs[i].axis('off')
plt.show()

print('############################################# Sobel ################################################')
fig, axs = plt.subplots(1, len(threshold_values), figsize=(20, 5))
for i, threshold in enumerate(threshold_values):
    print('################################'f' Threshold: {threshold} ' '##################################')
    # Computes precision and recall for the given threshold.
    binary_edges, (precision, recall) = adjust_threshold(image, sobel_operator, manual_edges, threshold)
    print(f"Precision: {precision}, Recall: {recall}")
    print()

    axs[i].imshow(binary_edges, cmap='gray')
    axs[i].set_title(f'Threshold: {threshold}\nPrecision: {precision:.2f}, Recall: {recall:.2f}')
    axs[i].axis('off')
plt.show()

print('############################################ Prewitt ###############################################')
fig, axs = plt.subplots(1, len(threshold_values), figsize=(20, 5))
for i, threshold in enumerate(threshold_values):
    print('################################'f' Threshold: {threshold} ' '##################################')
    # Computes precision and recall for the given threshold.
    binary_edges, (precision, recall) = adjust_threshold(image, prewitt_operator, manual_edges, threshold)
    print(f"Precision: {precision}, Recall: {recall}")
    print()

    axs[i].imshow(binary_edges, cmap='gray')
    axs[i].set_title(f'Threshold: {threshold}\nPrecision: {precision:.2f}, Recall: {recall:.2f}')
    axs[i].axis('off')
plt.show()
