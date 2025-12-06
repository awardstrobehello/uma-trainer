import cv2
import numpy as np
from mss import mss
from PIL import Image
import pyautogui
import time
import random
# Sometimes template matching returns duplicate matches, false positives that are a few pixels off from the actual match    
# This function removes them
def remove_duplicates(matches, pixel_distance=5):
    if len(matches) == 0:
        return []
    
    filtered = []
    for match in matches:
        is_duplicate = False
        for existing in filtered:
            dx = abs(match[0] - existing[0])
            dy = abs(match[1] - existing[1])
            if dx < pixel_distance and dy < pixel_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(match)
    
    return filtered


# Global variable to store selected monitor (set once at startup)
# at least until I add a GUI
SELECTED_MONITOR: dict | None = None


def list_monitors() -> list[dict]:
    # List all available physical monitors
    # might be reused for gui 
    # or in terminal if one accidentally picks the wrong one and wants to fix it
    
    sct = mss()
    monitors = []
    # Skip index 0 (combined view), start from index 1 (first physical monitor)
    for i, monitor in enumerate(sct.monitors[1:], start=1):
        monitors.append({
            'index': i,
            'name': f'Monitor {i}',
            'width': monitor['width'],
            'height': monitor['height'],
            'monitor': monitor
        })
    return monitors


def select_monitor(monitor_index: int | None = None) -> dict:
    # selection's stored globally for reuse
    
    global SELECTED_MONITOR
    
    monitors = list_monitors()
    
    if len(monitors) == 0:
        raise RuntimeError("No monitors detected??")
    
    if monitor_index is None:
        print("Choose a monitor")
        for mon in monitors:
            print(f"  [{mon['index']}] {mon['name']}: {mon['width']}x{mon['height']}")
        
        while True:
            try:
                choice = input(f"\nSelect monitor (1-{len(monitors)}): ").strip()
                monitor_index = int(choice)
                if 1 <= monitor_index <= len(monitors):
                    break
                else:
                    print(f"Enter a number between 1 and {len(monitors)}")
            except ValueError:
                print("Enter a valid number")
    else:
        if monitor_index < 1 or monitor_index > len(monitors):
            print(f"Invalid monitor index {monitor_index}. Using monitor 1 (first physical monitor).")
            monitor_index = 1
    
    selected = next(mon for mon in monitors if mon['index'] == monitor_index)
    SELECTED_MONITOR = selected
    print(f"Selected: {selected['name']} ({selected['width']}x{selected['height']})")
    return selected


def capture_screen(save_debug: bool = False) -> np.ndarray:
    global SELECTED_MONITOR
    
    if SELECTED_MONITOR is None:
        raise RuntimeError("No monitor selected!! call select_monitor() first")
    
    sct = mss()
    # grab() is memory only. use shot() for debugging
    screenshot_pil = Image.frombytes("RGB", 
                                     (SELECTED_MONITOR['monitor']['width'], SELECTED_MONITOR['monitor']['height']), 
                                     sct.grab(SELECTED_MONITOR['monitor']).rgb)
    
    # Convert PIL to numpy array, then BGR for OpenCV
    screenshot_np = np.array(screenshot_pil)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    
    if save_debug:
        cv2.imwrite('test_images/screenshot.png', screenshot_bgr)
    
    return screenshot_bgr



def get_search_region(screen_width: int, screen_height: int, region_type: str) -> dict:
    # returns a dict with the coordinates of the specified region
    
    if region_type not in SEARCH_REGIONS:
        available = ", ".join(SEARCH_REGIONS.keys())
        raise ValueError(f"Invalid region_type: '{region_type}'. Available: {available}")
    
    return SEARCH_REGIONS[region_type](screen_width, screen_height)


def crop_region(screenshot: np.ndarray, region: dict) -> np.ndarray:
    # Crop the screenshot to the specified region
    
    return screenshot[
        region['y']:region['y'] + region['height'],
        region['x']:region['x'] + region['width']
    ]


def debug_search_area(screenshot: np.ndarray, region: dict) -> None:
    # draws a rectangle on screenshot showing the search region and save it
    
    cv2.rectangle(
        screenshot,
        (region['x'], region['y']),
        (region['x'] + region['width'], region['y'] + region['height']),
        (0, 0, 0), 
        2
    )
    cv2.imwrite('test_images/region_visualization.png', screenshot)


def scale_template(template: np.ndarray, screen_width: int, screen_height: int, base_width: int = 1920, base_height: int = 1080) -> np.ndarray:
    # scales template image to match current screen resolution
    
    # Calculate scale factor based on width ratio (assuming the same aspect ratio)
    scale_factor = screen_width / base_width
    
    h, w = template.shape[:2]
    
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)
    
    # Resize template
    scaled_template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return scaled_template

def click_at_position(x: int, y: int) -> None:
    # global coordinates
    random_delay = random.uniform(0.1, 0.325)
    time.sleep(random_delay)
    pyautogui.moveTo(x, y, duration=0.225)
    pyautogui.click()


def click_template_match(match: tuple[int, int], template: np.ndarray, region: dict, click_center: bool = True) -> None:
    relative_x, relative_y = match
    
    if click_center:
        # Calculate center of template match
        template_h, template_w = template.shape[:2]
        relative_x = relative_x + template_w // 2
        relative_y = relative_y + template_h // 2
    
    global_x = region['x'] + relative_x
    global_y = region['y'] + relative_y

    click_at_position(global_x, global_y)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already grayscale


def find_template_in_region(template_path: str, screenshot: np.ndarray, region: dict, screen_width: int, screen_height: int, confidence_threshold: float = 0.7, use_grayscale: bool = False) -> tuple[list[tuple[int, int]], np.ndarray]:

    cropped_region = crop_region(screenshot, region)
    
    target_icon = cv2.imread(template_path)
    if target_icon is None:
        raise FileNotFoundError(f"Could not load template: {template_path}")
    
    template = scale_template(target_icon, screen_width, screen_height)
    original_template = template.copy()  # Keep original for return value
    
    # Convert to grayscale if requested (for better matching when colors vary)
    if use_grayscale:
        cropped_region_gray = convert_to_grayscale(cropped_region)
        template_gray = convert_to_grayscale(template)
        result = cv2.matchTemplate(cropped_region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    else:
        result = cv2.matchTemplate(cropped_region, template, cv2.TM_CCOEFF_NORMED)
    match_locations = np.where(result >= confidence_threshold)
    matches = list(zip(*match_locations[::-1]))  # Convert to (x, y) format
    matches = remove_duplicates(matches)
    if "support_card_type" in template_path or "unity_training" in template_path:
        return matches, original_template
    elif len(matches) == 0:
        print(f"Max confidence was not enough: {result.max():.5f}")
    #     print(f"Min confidence: {result.min():.5f}")
    
    return matches, original_template

# Registry of search region definitions relative to screen dimensions
# aka. DYNAMIC SEARCH REGIONS !!!!! 
# Each function takes (width, height) 
# returning a dict with x, y, width, height
# since it's gonna be used all the time maybe a class would be better?
# nah
SEARCH_REGIONS = {
    "support_region": lambda w, h: {
        'x': int(w * 0.418),
        'y': int(h * 0.13),
        'width': int(w * 0.08),
        'height': int(h * 0.6)
    },
    "training_region": lambda w, h: {
        'x': int(w * 0.135),
        'y': int(h * 0.703),
        'width': int(w * 0.3),
        'height': int(h * 0.2)
    },
    "choice_region": lambda w, h: { 
        'x': int(w * 0.135),
        'y': int(h * 0.71),
        'width': int(w * 0.3),
        'height': int(h * 0.22)
    },
}

FRIENDSHIP_LEVEL = {
    "gray": [120, 108, 110],
    "blue": [255, 192, 42],
    "green": [30, 230, 162],
    "yellow": [30, 173, 255],
    "max": [120, 235, 255],
}

TRAINING_AREA = {
    "speed": 'assets/icons/train_spd.png',
    "stamina": "assets/icons/train_sta.png",
    "power": "assets/icons/train_pow.png",
    "guts": "assets/icons/train_guts.png",
    "wit": "assets/icons/train_wit.png",
}

SUPPORT_CARD_TYPES = {
    "speed": "assets/icons/support_card_type_spd.png",
    "stamina": "assets/icons/support_card_type_sta.png",
    "power": "assets/icons/support_card_type_pow.png",
    "guts": "assets/icons/support_card_type_guts.png",
    "wit": "assets/icons/support_card_type_wit.png",
    "friend": "assets/icons/support_card_type_friend.png",
}

UNITY_TRAINING_TYPES = {
    "expired": "assets/icons/unity_expired_burst.png",
    "training": "assets/icons/unity_training.png",
    "burst": "assets/icons/unity_spirit_burst.png",
}



def get_pixel_color(screenshot: np.ndarray, x: int, y: int) -> np.ndarray:
    # Get BGR color of a pixel at global coordinates
    return screenshot[y, x]  # OpenCV arrays are [height, width], so [y, x]


def find_friendship_level(icon_match: tuple[int, int], template: np.ndarray, screenshot: np.ndarray, region: dict, screen_height: int, base_height: int = 1080) -> str:
    # Reads color of the friendship bar to find the friendship level
    # icon_match is (x, y) - top-left corner of match relative to cropped region
    # template is the scaled template image to get width/height
    
    icon_x, icon_y = icon_match
    template_h, template_w = template.shape[:2]
    
    # Calculate icon center relative to cropped region
    icon_center_x = icon_x + template_w // 2
    icon_center_y = icon_y + template_h // 2
    
    # Calculate friendship bar position offset (scales with resolution)
    # Reference uses 66 pixels at 1080p
    icon_to_friend_bar_distance = int(66 * (screen_height / base_height))
    
    # Convert to global coordinates
    friendship_bar_x = region['x'] + icon_center_x
    friendship_bar_y = region['y'] + icon_center_y + icon_to_friend_bar_distance
    
    color_bgr = get_pixel_color(screenshot, friendship_bar_x, friendship_bar_y)
    closest_color = min(FRIENDSHIP_LEVEL.items(), 
                       key=lambda item: np.linalg.norm(np.array(item[1]) - color_bgr))
    
    return closest_color[0]

def main():
    select_monitor(monitor_index=None)    
    screenshot = capture_screen(save_debug=True)

    screen_width, screen_height = screenshot.shape[1], screenshot.shape[0]
    # TODO: one run-through scan and calculation for training

    # State check. in "choice" area? rest, infirmary, etc
    region_type = "choice_region"
    region_dict = get_search_region(screen_width, screen_height, region_type)
    cropped_region = crop_region(screenshot, region_dict)
    # debug_search_area(screenshot.copy(), region_dict)

    # Template matching for training button
    matches, template = find_template_in_region('assets/buttons/training_btn.png', screenshot, region_dict, screen_width, screen_height)
    if len(matches) > 0:
        print("Training button found")
        click_template_match(matches[0], template, region_dict, click_center=True)
    
    # Training area detection
    time.sleep(random.uniform(0.4, 0.5))

    # debug_search_area(screenshot.copy(), region_dict)

    prev_value = 'speed' # this is the default value. if it's the same as the current value, don't click
    # TODO: make it click on speed first so there aren't any issues with matching
    training_locations = {}
    
    # In practice this only needs to be run once in the entire run, just to get the locations
    # so training_locations will need to be a global variable
    for name, template_path in TRAINING_AREA.items():
        screenshot = capture_screen(save_debug=True)
        region_type = "training_region"
        region_dict = get_search_region(screen_width, screen_height, region_type)
        cropped_region = crop_region(screenshot, region_dict)

        # Use grayscale for training icons since they have different colors based on level
        matches, template = find_template_in_region(template_path, screenshot, region_dict, screen_width, screen_height, use_grayscale=True)
        count = 0

        if len(matches) == 0:
            while len(matches) == 0: # sometimes it takes a few tries to find the training type
                screenshot = capture_screen(save_debug=True)
                region_type = "training_region"
                region_dict = get_search_region(screen_width, screen_height, region_type)
                cropped_region = crop_region(screenshot, region_dict)

                # Use grayscale for training icons since they have different colors based on level
                matches, template = find_template_in_region(template_path, screenshot, region_dict, screen_width, screen_height, use_grayscale=True)
                count += 1
                if count > 10:
                    raise RuntimeError(f"Training type {name} not found after 10 attempts")
        training_locations[name] = (matches, template)
        print(f"Training type {name} found at: {matches}")


    # Support is in the same screenshot as training, so we don't need to capture a new one
    region_type = "support_region"
    region_dict = get_search_region(screen_width, screen_height, region_type)
    cropped_region = crop_region(screenshot, region_dict)

    # The loop here will be run for each training type
    # TODO: create spirit burst checker as well
    for name, template_path in SUPPORT_CARD_TYPES.items():
        matches, template = find_template_in_region(template_path, screenshot, region_dict, screen_width, screen_height)
        for match in matches:
            friendship_level = find_friendship_level(match, template, screenshot, region_dict, screen_height)
            print(f"Found {name} icon at: {match} with friendship level {friendship_level}")


    # Collect all matches first and then filter overlaps
    unity_matches = {}
    unity_templates = {}

    debug_search_area(screenshot.copy(), region_dict)

    for name, template_path in UNITY_TRAINING_TYPES.items():
        matches, template = find_template_in_region(template_path, screenshot, region_dict, screen_width, screen_height, use_grayscale=True)
        unity_matches[name] = matches
        unity_templates[name] = template
        print(f"Found {len(matches)} {name} unity icons at: {matches}")


    # Remove "training" icons that overlap with "expired" icons (they're the same position)
    if "expired" in unity_matches and "training" in unity_matches:
        expired_positions = unity_matches["expired"]
        training_positions = unity_matches["training"]
        
        # filter out training icons that're in the same position as expired icons
        # Using pixel_distance to account for slight variations
        filtered_training = []
        for training_pos in training_positions:
            is_overlapping = False
            for expired_pos in expired_positions:
                dx = abs(training_pos[0] - expired_pos[0])
                dy = abs(training_pos[1] - expired_pos[1])
                # If within 10 pixels consider it overlapping (same icon)
                if dx < 10 and dy < 10:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_training.append(training_pos)
        
        unity_matches["training"] = filtered_training
        print(f"After removing overlaps: {len(filtered_training)} training unity icons remain")
    for name, matches in unity_matches.items():
        for match in matches:
            print(f"Found {name} icon at: {match}")

if __name__ == "__main__":
    main()