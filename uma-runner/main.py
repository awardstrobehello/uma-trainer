import cv2
import numpy as np
from mss import mss
from PIL import Image

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

def main():
    select_monitor(monitor_index=None)    
    screenshot = capture_screen(save_debug=True)

    # TODO: create loop that selects region and runs template matching
    # TODO: create new support images so size doesn't need to be altered
        # especially given that I'm going to make new images for 16:9 

    screen_width, screen_height = screenshot.shape[1], screenshot.shape[0]
    region_type = "choice_region"
    region_dict = get_search_region(screen_width, screen_height, region_type)

    # DEBUG visualizes the search region with a box
    debug_search_area(screenshot.copy(), region_dict)

    cropped_region = crop_region(screenshot, region_dict)
    
    # Load and prepare template for matching
    image_path = 'assets/icons/support_card_type_spd.png'
    target_icon = cv2.imread(image_path)
    if target_icon is None:
        raise FileNotFoundError(f"Could not load target icon: {image_path}")
    
    h, w = target_icon.shape[:2]

    # temporary fix for icon detection - 1920x1080 to 2560x1440
    template = cv2.resize(target_icon, (int(w * 1.333), int(h * 1.333)))

    # Template matching on cropped region
    result = cv2.matchTemplate(cropped_region, template, cv2.TM_CCOEFF_NORMED)
    confidence_threshold = 0.9
    match_locations = np.where(result >= confidence_threshold)
    
    matches = list(zip(*match_locations[::-1])) # converting match locations to (x, y) format

    print(f"Max confidence: {result.max():.5f}")
    print(f"Min confidence: {result.min():.5f}")
    
    matches = remove_duplicates(matches)

    print(f"Found {len(matches)} icons at: {matches}")

if __name__ == "__main__":
    main()