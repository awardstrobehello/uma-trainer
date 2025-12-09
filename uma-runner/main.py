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


def draw_rectangle_at_point(screenshot: np.ndarray, x: int, y: int, size: int = 10, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
    # x, y: Global coordinates
    # thickness = line thickness
    
    cv2.rectangle(
        screenshot,
        (x - size, y - size),
        (x + size, y + size),
        color,
        thickness
    )
    # Also draw crosshair for better visibility
    cv2.line(screenshot, (x - size * 2, y), (x + size * 2, y), color, 1)
    cv2.line(screenshot, (x, y - size * 2), (x, y + size * 2), color, 1)


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


def find_template_in_region(template_path: str, screenshot: np.ndarray, region: dict, screen_width: int, screen_height: int, confidence_threshold: float = 0.8, use_grayscale: bool = False) -> tuple[list[tuple[int, int]], np.ndarray]:

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
    # elif len(matches) == 0:
    #     print(f"Max confidence was not enough: {result.max():.5f}")
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
    "energy_region": lambda w, h: {
        'x': int(w * 0.227),
        'y': int(h * 0.11),
        'width': int(w * 0.2),
        'height': int(h * 0.03)
    },
}

FRIENDSHIP_LEVEL = {
    "gray": [120, 108, 110],
    "blue": [255, 192, 42],
    "green": [30, 230, 162],
    "orange": [30, 173, 255],
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
    "training": "assets/icons/unity_training.png",
    "burst": "assets/icons/unity_spirit_burst.png",
    # "expired": "assets/icons/unity_expired_burst.png",
}

# base = total_supports + (non_max_friends * 0.5) + hint_bonus
# speed and wit are usually heaviest - you need speed to win, wit to do more training
# priority_bonus = 1 + PRIORITY_WEIGHT
# controls how much priority stats affect the training score adder
PRIORITY_WEIGHT = {
    "HEAVY": 0.75,   # Strong priority influence
    "MEDIUM": 0.5,   # Moderate priority influence
    "SMALL": 0.25,   # Weak priority influence
    "NONE": 0.0      # No priority influence
}

# how much each stat/resource contributes to choice scoring
CHOICE_WEIGHT = {
    "spd": 1.0,        # Speed stat weight
    "sta": 1.0,        # Stamina stat weight
    "pwr": 1.0,        # Power stat weight
    "guts": 1.0,       # Guts stat weight
    "wit": 1.0,        # Wit stat weight
    "hp": 0.5,         # Energy (HP) weight (lower because overflow penalty applies)
    "mood": 0.8,       # Mood weight (only counts if mood < max)
    "max_energy": 0.3, # Max energy increase weight
    "skillpts": 0.4,   # Skill points weight
    "bond": 0.6,       # Friendship bond weight
}

# Training decision constants
# HINT_POINT = 0.6  # I ignore hints in my runs so I don't need this right now
NON_MAX_FRIEND_INCREASE = 0.25  # increases score for each non-maxed friendship level


def get_pixel_color(screenshot: np.ndarray, x: int, y: int) -> np.ndarray:
    # Get BGR color of a pixel at global coordinates
    return screenshot[y, x]  # OpenCV arrays are [height, width], so [y, x]


def get_color_below_match(match: tuple[int, int], template: np.ndarray, screenshot: np.ndarray, region: dict, screen_height: int, pixels_down: int = 60, pixels_right: int = 0, base_height: int = 1080, draw_rectangle: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    # Get BGR color of a pixel at a specified distance below
    # or optionally right of a template match
    
    match_x_relative, match_y_relative = match
    template_h, template_w = template.shape[:2]
    
    # Calculate center of template match relative to cropped region
    match_center_x_relative = match_x_relative + template_w // 2
    match_center_y_relative = match_y_relative + template_h // 2
    
    # Scale the pixel distances based on resolution
    scaled_pixels_down = int(pixels_down * (screen_height / base_height))
    scaled_pixels_right = int(pixels_right * (screen_height / base_height))
    
    # Calculate position with offset
    check_x_relative = match_center_x_relative + scaled_pixels_right
    check_y_relative = match_center_y_relative + scaled_pixels_down
    
    # Convert to global coordinates
    global_x = region['x'] + check_x_relative
    global_y = region['y'] + check_y_relative
    
    # Draw rectangle at check location if requested
    if draw_rectangle:
        # Draw a small rectangle (5x5 pixels) around the check point
        rect_size = 5
        cv2.rectangle(
            screenshot,
            (global_x - rect_size, global_y - rect_size),
            (global_x + rect_size, global_y + rect_size),
            (0, 255, 0),  # Green color in BGR
            2  # Thickness
        )
        # Also draw a crosshair for better visibility
        cv2.line(screenshot, (global_x - 10, global_y), (global_x + 10, global_y), (0, 255, 0), 1)
        cv2.line(screenshot, (global_x, global_y - 10), (global_x, global_y + 10), (0, 255, 0), 1)
    
    # Get color at that position
    color_bgr = get_pixel_color(screenshot, global_x, global_y)
    return color_bgr, (global_x, global_y)


def count_pixels_of_color(screenshot: np.ndarray, region: dict, target_color_bgr: list[int], tolerance: int = 2) -> int:
    # Count pixels in a region that match the target color (BGR format)
    # target_color_bgr: [B, G, R] color to match
    # tolerance: ±tolerance for color matching
    
    cropped = crop_region(screenshot, region)
    
    # Convert target color to numpy array (BGR format)
    color = np.array(target_color_bgr, dtype=np.uint8)
    
    # Define min/max range with tolerance
    color_min = np.clip(color - tolerance, 0, 255)
    color_max = np.clip(color + tolerance, 0, 255)
    
    # Create mask for pixels within color range
    mask = cv2.inRange(cropped, color_min, color_max)
    
    # Count non-zero pixels (matching pixels)
    pixel_count = cv2.countNonZero(mask)
    return pixel_count


def check_energy_level(screenshot: np.ndarray, screen_width: int, screen_height: int, threshold: float = 0.85) -> tuple[float, float] | tuple[int, int]:
    # Detects energy level by finding the right end of the energy bar and counting empty pixels
    # Returns: (current_energy, max_energy) or (-1, -1) if detection fails
    # 
    # Method: Template match to find bar end → count gray pixels (empty energy) → calculate percentage
    
    region_type = "energy_region"
    region_dict = get_search_region(screen_width, screen_height, region_type)
    debug_search_area(screenshot.copy(), region_dict)



    # Find the right end of the energy bar using template matching
    # Try primary template first
    matches, template = find_template_in_region(
        "assets/ui/energy_bar_right_end_part.png",
        screenshot,
        region_dict,
        screen_width,
        screen_height,
        confidence_threshold=threshold
    )
    
    # If primary template fails, try alternative (different bar roundness)
    if len(matches) == 0:
        matches, template = find_template_in_region(
            "assets/ui/energy_bar_right_end_part_2.png",
            screenshot,
            region_dict,
            screen_width,
            screen_height,
            confidence_threshold=threshold
        )
    
    if len(matches) == 0:
        print("Warning: Could not find energy bar right end")
        return -1, -1
    
    # Get template dimensions (already scaled to current resolution)
    template_h, template_w = template.shape[:2]
    
    # Get the x position of the right end (first match)
    # The match gives us the top-left corner, so right end is at x + template_width
    right_end_x_relative, right_end_y_relative = matches[0]
    right_end_x_relative = right_end_x_relative + template_w  # Actual right edge position
    energy_bar_length = right_end_x_relative  # Length from left edge of region to right end
    
    # Calculate the middle vertical position of the energy bar for pixel counting
    bar_middle_y_relative = right_end_y_relative + template_h // 2
    
    # Create region for counting empty energy pixels (gray color)
    # Count from left edge of energy region to the right end
    empty_energy_region = {
        'x': region_dict['x'],
        'y': region_dict['y'] + bar_middle_y_relative,
        'width': energy_bar_length,
        'height': 1  # Single pixel row for counting
    }
    
    empty_energy_color_bgr = [117, 117, 117]
    empty_energy_pixel_count = count_pixels_of_color(
        screenshot,
        empty_energy_region,
        empty_energy_color_bgr,
        tolerance=2
    )
    
    # Calculate energy level
    # Reference: 236 pixels = 100 energy at 1080p (scales with resolution)
    base_resolution_height = 1080
    hundred_energy_pixel_constant = int(236 * (screen_height / base_resolution_height))
    
    total_energy_length = energy_bar_length - 1  # Subtract edge pixel
    filled_pixels = total_energy_length - empty_energy_pixel_count
    
    if hundred_energy_pixel_constant > 0:
        energy_level = (filled_pixels / hundred_energy_pixel_constant) * 100
        max_energy = (total_energy_length / hundred_energy_pixel_constant) * 100
    else:
        return -1, -1
    
    print(f"Energy: {energy_level:.1f}% / {max_energy:.1f}% (bar length: {total_energy_length}px, empty: {empty_energy_pixel_count}px)")
    
    return energy_level, max_energy


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


class TrainingLocations:
#     Stores training type locations detected once at startup.
#     So re-detecting them is unnecessary
    
    def __init__(self):
        self.locations: dict[str, tuple[list[tuple[int, int]], np.ndarray]] = {}
        self.region_dict: dict | None = None
        self._initialized = False
    
    def detect_all(self, screen_width: int, screen_height: int) -> None:
        # Called once at startup to detect all training type locations
        # Converts relative coordinates to global coordinates and stores them

        region_type = "training_region"
        self.region_dict = get_search_region(screen_width, screen_height, region_type)
        time.sleep(0.1)
        print("Training types: ", end="")
        for name, template_path in TRAINING_AREA.items():
            screenshot = capture_screen(save_debug=True)
            
            # Grayscale for training icons - different colors based on level
            matches, template = find_template_in_region(template_path, screenshot, self.region_dict, 
                                                       screen_width, screen_height, use_grayscale=True)
            count = 0

            if len(matches) == 0:
                while len(matches) == 0:  # sometimes it takes a few tries to find the training type
                    screenshot = capture_screen(save_debug=False)
                    matches, template = find_template_in_region(template_path, screenshot, self.region_dict, 
                                                               screen_width, screen_height, use_grayscale=True)
                    count += 1
                    if count > 10:
                        raise RuntimeError(f"Training type {name} not found after 10 attempts")
            
            # Convert relative coordinates to global coordinates (center of template)
            template_h, template_w = template.shape[:2]
            global_matches = []
            for relative_x, relative_y in matches:
                # Calculate center of template match
                relative_x = relative_x + template_w // 2
                relative_y = relative_y + template_h // 2
                
                # Convert to global coordinates
                global_x = self.region_dict['x'] + relative_x
                global_y = self.region_dict['y'] + relative_y
                global_matches.append((global_x, global_y))
            
            self.locations[name] = (global_matches, template)
            print(f"{name}", end=" ")
            # print(f"Locations: {global_matches}")

        print("found")
        self._initialized = True
    
    def get_location(self, training_type: str) -> tuple[list[tuple[int, int]], np.ndarray] | None:
        # Returns stored location (global coordinates, center of template) and template for a training type
        if not self._initialized:
            raise RuntimeError("TrainingLocations not initialized. Call detect_all() first.")
        return self.locations.get(training_type)


def drag_through_training_types_and_get_score(training_locations: TrainingLocations, screen_width: int, screen_height: int) -> dict:
    # Drags through all training types by clicking and holding on the leftmost training type
    # and dragging through all training types in order (left to right)
    
    if not training_locations._initialized:
        raise RuntimeError("TrainingLocations not initialized. Call detect_all() first.")
    
    # Collect all training type locations with their global coordinates
    all_locations: list[tuple[int, int, str]] = []  # (x, y, training_type)
    
    for training_type in TRAINING_AREA.keys():
        location_data = training_locations.get_location(training_type)
        if location_data is not None:
            global_matches, _ = location_data
            if len(global_matches) > 0:
                # Use the first match for each training type (assuming one per type)
                x, y = global_matches[0]
                all_locations.append((x, y, training_type))
    
    print("--------------------------------")


    if len(all_locations) == 0:
        print("No training type locations found")
        return
    
    # Sort by x-coordinate to get left-to-right order
    all_locations.sort(key=lambda loc: loc[0])
    
    # Get the leftmost location (start position)
    start_x, start_y, start_type = all_locations[0]
    print(f"Starting drag at {start_type}: ({start_x}, {start_y})")
       
    # Move to start position
    pyautogui.moveTo(start_x, start_y, duration=0.225)
    
    # Begin click hold
    pyautogui.mouseDown()
    
    # Create dict for calculating training score
    training_score_dict = {
        "speed": 0,
        "stamina": 0,
        "power": 0,
        "guts": 0,
        "wit": 1, # because the extra turn is that valuable
    }

    print("--------------------------------")
    # Drag through all remaining locations
    for x, y, training_type in all_locations:
        # TODO: check infirmary to see if it's needed
        pyautogui.moveTo(x, y, duration=0.001)
        print(f"Training type {training_type}")

        screenshot = capture_screen(save_debug=True)
        region_type = "support_region"
        region_dict = get_search_region(screen_width, screen_height, region_type)
        cropped_region = crop_region(screenshot, region_dict)
        # debug_search_area(screenshot.copy(), region_dict)

        # Support detection loop
        for name, template_path in SUPPORT_CARD_TYPES.items():
            matches, template = find_template_in_region(template_path, screenshot, region_dict, screen_width, screen_height)
            for match in matches:
                friendship_level = find_friendship_level(match, template, screenshot, region_dict, screen_height)
                print(f"| {name}", end=" ")

                # Non-maxed friendship bonus
                if friendship_level not in ("max", "orange"):
                    training_score_dict[training_type] += NON_MAX_FRIEND_INCREASE
                    print(f"; {friendship_level}; {NON_MAX_FRIEND_INCREASE}", end=" ")

                # Rainbow bonus
                if name == training_type and (friendship_level in ("max", "orange")):
                    training_score_dict[training_type] += 2.2
                    print(f"; rainbow + 2.2", end=" ")
                print()

                # print(f"Location: {match}")

        # Unity training types detection
        unity_matches = {}
        unity_templates = {}
       
        for name, template_path in UNITY_TRAINING_TYPES.items():
            matches, template = find_template_in_region(template_path, screenshot, region_dict, screen_width, screen_height, use_grayscale=False)
            unity_matches[name] = matches
            # print(f"{len(matches)} {name}", end=" ")
            for match in matches: 
                if name in "burst":
                    training_score_dict[training_type] += 2
                    print(f"burst (2)", end="\n")
                elif name in "training":
                    color_below, check_coords = get_color_below_match(
                        match, template, screenshot, region_dict, screen_height, 
                        pixels_down=55, 
                        pixels_right=0,
                        draw_rectangle=False
                    )
                    # print(f"{name}, position at {check_coords}: BGR{color_below.tolist()}", end=" | ")
                    if color_below[2] < 100:
                        training_score_dict[training_type] += 0
                        print(f"| | expired (0)", end="\n")
                    else:
                        training_score_dict[training_type] += 1
                        print(f"| | training (1)", end="\n")
            
            # print(f"{len(matches)} {name} unity icons at: {matches}")

        # Save screenshot with color check rectangles for debugging
        cv2.imwrite('test_images/color_check_locations.png', screenshot)
        
        print("--------------------------------")

    
    # Release mouse button
    pyautogui.mouseUp()
    print("Completed drag")
    print("--------------------------------")

    return training_score_dict
        
def main():
    select_monitor(monitor_index=None)    
    screenshot = capture_screen(save_debug=True)

    screen_width, screen_height = screenshot.shape[1], screenshot.shape[0]

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
    else:
        print("Training button not found")



    # Detect all training locations once - only needs to be called once per run
    training_locations = TrainingLocations()
    training_locations.detect_all(screen_width, screen_height)
    training_score_dict = drag_through_training_types_and_get_score(training_locations, screen_width, screen_height)

    # returns energy level and max energy level
    energy_level, max_energy = check_energy_level(screenshot, screen_width, screen_height)
    # Show training score
    for name, score in training_score_dict.items():
            print(f"{name}: {score}")
    #TODO: Check energy level before decision


if __name__ == "__main__":
    main()