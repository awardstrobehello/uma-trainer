import cv2
import numpy as np
from mss import mss
from PIL import Image

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


def main():
    mss().shot(output='test_images/screenshot.png')

    screenshot = cv2.imread('test_images/screenshot.png')

    # assuming 16:9 aspect ratio
    screen_width, screen_height = screenshot.shape[1], screenshot.shape[0]
    training_region = {
        'x': int(screen_width * 0.418),
        'y': int(screen_height * 0.13),
        'width': int(screen_width * 0.08),
        'height': int(screen_height * 0.6)
    }  

    # DEBUG to check region where it's looking 
    # cv2.rectangle(
    #     screenshot,
    #     (training_region['x'], training_region['y']),  # top left corner
    #     (training_region['x'] + training_region['width'], 
    #     training_region['y'] + training_region['height']),  # bottom right corner
    #     (0, 0, 0),  # black
    #     2  # line thickness
    # )
    # cv2.imwrite('test_images/region_visualization.png', screenshot)


    cropped = screenshot[
        training_region['y']:training_region['y']+training_region['height'],
        training_region['x']:training_region['x']+training_region['width']
    ]

    target_icon = cv2.imread('assets/icons/support_card_type_spd.png')

    result = cv2.matchTemplate(cropped, target_icon, cv2.TM_CCOEFF_NORMED)
    threshold = 0.69 # unironically the best confidence threshold
    match_locations = np.where(result >= threshold)

    matches = []
    for pt in zip(*match_locations[::-1]):
        matches.append(pt)

    for match in matches:
        match = (match[0] + training_region['x'], match[1] + training_region['y'])

    print(f"Max confidence: {result.max():.5f}")
    print(f"Min confidence: {result.min():.5f}")
    
    matches = remove_duplicates(matches)

    print(f"Found {len(matches)} icons at: {matches}")

if __name__ == "__main__":
    main()