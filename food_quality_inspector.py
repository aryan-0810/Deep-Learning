import os
import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def safe_imshow(window_name, image):
    """Display an image safely without freezing the console menu loop."""
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
    except cv2.error:
        print(f"[{window_name}] Image display is unavailable in this environment; continuing without showing the window.")


def close_all_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def list_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = []
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if os.path.isfile(full_path) and os.path.splitext(entry)[1].lower() in VALID_EXTENSIONS:
            file_names.append(entry)
    return file_names, base_dir


def select_image():
    files, base_dir = list_images()
    if not files:
        raise FileNotFoundError("No image files were found in the workspace. Please add a JPG/PNG image first.")

    print("\nAvailable images:")
    for index, file_name in enumerate(files, start=1):
        print(f"  {index}. {file_name}")

    user_input = input("Enter the image number or full path: ").strip()

    if user_input.isdigit():
        index = int(user_input) - 1
        if 0 <= index < len(files):
            return os.path.join(base_dir, files[index])
        raise ValueError("Invalid image number selected.")

    if os.path.exists(user_input):
        return user_input

    candidate = os.path.join(base_dir, user_input)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"Image '{user_input}' was not found.")


def load_and_inspect_image():
    image_path = select_image()
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load the image: {image_path}")

    print(f"\nImage loaded from: {image_path}")
    print(f"Height: {image.shape[0]} pixels")
    print(f"Width: {image.shape[1]} pixels")
    print(f"Channels: {image.shape[2]} ")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    safe_imshow("Original Image", image)
    safe_imshow("Grayscale Image", gray)

    save_name = os.path.splitext(os.path.basename(image_path))[0] + "_gray.jpg"
    save_path = os.path.join(os.path.dirname(image_path), save_name)
    success = cv2.imwrite(save_path, gray)
    if success:
        print(f"Grayscale image saved as: {save_path}")
    else:
        print("Warning: grayscale image could not be saved.")

    return image


def resize_and_analyse_channels(image):
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(resized)

    print("\nResized image: 256x256 using INTER_AREA")
    print(f"Blue channel average intensity: {b.mean():.2f}")
    print(f"Green channel average intensity: {g.mean():.2f}")
    print(f"Red channel average intensity: {r.mean():.2f}")

    safe_imshow("Blue Channel", b)
    safe_imshow("Green Channel", g)
    safe_imshow("Red Channel", r)

    return resized


def transformation_pipeline(image):
    angle = input("Enter rotation angle in degrees: ").strip()
    try:
        angle_value = float(angle)
    except ValueError:
        raise ValueError("Angle must be a valid number.")

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_value, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

    crop_h = int(rotated.shape[0] * 0.6)
    crop_w = int(rotated.shape[1] * 0.6)
    start_y = (rotated.shape[0] - crop_h) // 2
    start_x = (rotated.shape[1] - crop_w) // 2
    cropped = rotated[start_y:start_y + crop_h, start_x:start_x + crop_w]

    flipped = cv2.flip(cropped, 1)

    safe_imshow("Rotated Image", rotated)
    safe_imshow("Cropped Center Region", cropped)
    safe_imshow("Horizontally Flipped Image", flipped)

    return rotated, cropped, flipped


def edge_quality_scan(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    edge_pixels = cv2.countNonZero(edges)
    print(f"\nTotal edge pixels: {edge_pixels}")

    if edge_pixels > 5000:
        verdict = "High texture (good detail)"
    else:
        verdict = "Low texture (may need re-shoot)"

    print(f"Quality verdict: {verdict}")
    safe_imshow("Edge Map", edges)

    return edge_pixels, verdict


def main():
    current_image = None

    print("========================================================")
    print("             Food Delivery Image Quality Inspector")
    print("========================================================")

    while True:
        print("\nMenu:")
        print("1. Load & inspect image")
        print("2. Resize and analyse colour channels")
        print("3. Apply transformation pipeline")
        print("4. Run edge-based quality scan")
        print("5. Exit")

        choice = input("Select an option: ").strip()

        try:
            if choice == "1":
                current_image = load_and_inspect_image()
            elif choice == "2":
                if current_image is None:
                    print("Please load an image first using option 1.")
                    continue
                resize_and_analyse_channels(current_image)
            elif choice == "3":
                if current_image is None:
                    print("Please load an image first using option 1.")
                    continue
                transformation_pipeline(current_image)
            elif choice == "4":
                if current_image is None:
                    print("Please load an image first using option 1.")
                    continue
                edge_quality_scan(current_image)
            elif choice == "5":
                print("Exiting the food quality inspector. Goodbye!")
                close_all_windows()
                break
            else:
                print("Invalid menu option. Please choose from 1 to 5.")

        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")

        close_all_windows()


if __name__ == "__main__":
    main()
