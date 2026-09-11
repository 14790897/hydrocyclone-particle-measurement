import cv2
import numpy as np
from pathlib import Path

def enhance_contrast(image, alpha=2, beta=20):
    """
    Enhance image contrast.
    alpha: contrast (1.0-3.0); higher = more contrast
    beta: brightness (0-100), added to the whole image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_exposure(image, gamma=2):
    """
    Enhance exposure.
    gamma: exposure; <1.0 brightens, >1.0 darkens
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)

def enhance_contrast_CLAHE(image):
    """
    Contrast-limited adaptive histogram equalisation (CLAHE).
    Input: BGR colour or grayscale image
    Output: enhanced grayscale image
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    enhanced_image = clahe.apply(image)

    return enhanced_image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def sharpen_after_gaussian(image, kernel_size=(5, 5), sigma=1.0):
    smoothed = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(smoothed, -1, kernel)
    return image

def adjust_hsv_properties(
    image, saturation_scale=1.2, hue_shift=10, brightness_scale=1.2
):
    """
    Adjust saturation, hue and brightness.
    image: input BGR image
    saturation_scale: >1 boosts, <1 reduces saturation
    hue_shift: hue offset (-180 to 180)
    brightness_scale: >1 brightens, <1 darkens
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    s = cv2.multiply(s, saturation_scale)
    s = np.clip(s, 0, 255).astype(np.uint8)

    h = (h + hue_shift) % 180

    v = cv2.multiply(v, brightness_scale)
    v = np.clip(v, 0, 255).astype(np.uint8)

    hsv_adjusted = cv2.merge([h, s, v])

    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    return adjusted_image

def apply_s_curve_contrast(image, alpha=1.0, beta=0.5):
    """
    Enhance contrast with an S-curve.

    Args:
    - image: input image, BGR (OpenCV default).
    - alpha: S-curve steepness (default 1.0; higher = more contrast).
    - beta: curve inflection point (0-1, default 0.5).

    Returns:
    - result: processed image.
    """
    if len(image.shape) == 2:
        is_gray = True
        img = image.copy()
    else:
        is_gray = False
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_normalized = img / 255.0

    s_curve = 1 / (1 + np.exp(-alpha * (img_normalized - beta)))

    result = (s_curve * 255).astype(np.uint8)

    if not is_gray:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def unsharp_masking(img, ksize=5, sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened

def process_image(image_path, output_path, config=None):
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imdecode(
        np.fromfile(file=str(Path(image_path)), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_original = image.copy()
    # image = enhance_contrast_CLAHE(image)
    # image = sharpen_after_gaussian(image, kernel_size=(5, 5), sigma=1.0)
    # image = unsharp_masking(image)
    image = enhance_contrast(image)
    # image = sharpen_image(image)
    # image = enhance_exposure(image, gamma=2)
    # image = adjust_hsv_properties(
    #     image, saturation_scale=3, hue_shift=15, brightness_scale=1.1
    # )
    # image = apply_s_curve_contrast(image, alpha=8.0, beta=0.5)

    image_show = np.hstack((image_original, image))

    # cv2.imshow("Processed Image", image_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    process_image("1.jpg", "denoised_sharpened_image.jpg")
