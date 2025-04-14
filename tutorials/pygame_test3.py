import pygame
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Set up display
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Move PNG Example")

# Load the background
background = pygame.Surface((width, height))
background.fill((0, 0, 0))  # Black background

# Load images
image_names = ['1_OgunL.png', '2_VenusL.png', '3_MolliL.png', '3_NauvisL.png', '6_JupiterL.png']
images = []
image_rects = []

for name in image_names:
    try:
        img = pygame.image.load(f'./pictures/Calidus1/planets/{name}')
        images.append(img)
        image_rects.append(img.get_rect())
    except pygame.error as e:
        print(f"Error loading image {name}: {e}")

# Pre-computed transformations
num_images = len(images)
orbit_radius = 150

# Set initial positions for images
for i, rect in enumerate(image_rects):
    rect.center = (width // 2, height // 2)  # Start at the center

# Create small objects
num_small_objects = 10
small_objects = [pygame.Surface((np.random.randint(1, 6), np.random.randint(1, 6))) for _ in range(num_small_objects)]
for obj in small_objects:
    obj.fill((255, 255, 255))  # White small objects

# Set initial positions for small objects
small_object_rects = [obj.get_rect(center=(np.random.randint(0, width), np.random.randint(0, height))) for obj in small_objects]

# Main loop
WRITE = 1  # Set to True to write to file
FPS = 60
clock = pygame.time.Clock()

frame_count = 0
while frame_count < 181:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear the screen
    screen.blit(background, (0, 0))

    # Update and draw images
    for i, rect in enumerate(image_rects):
        # Calculate new position for orbiting
        angle = frame_count * 0.01 + (i * (2 * np.pi / num_images))  # Increment angle over time
        rect.x = (width // 2) + orbit_radius * np.cos(angle) - rect.width // 2
        rect.y = (height // 2) + orbit_radius * np.sin(angle) - rect.height // 2

        # Draw the image with transformations
        scaled_image = pygame.transform.scale(images[i], (int(images[i].get_width() * 0.5), int(images[i].get_height() * 0.5)))  # Scale
        screen.blit(scaled_image, rect)

    # Draw small objects
    for small_rect in small_object_rects:
        screen.blit(small_objects[0], small_rect)  # Draw small objects

    # Update the display
    pygame.display.flip()

    # Save frame to video if WRITE is True
    if WRITE:
        pygame.image.save(screen, f"./pygame_frames/frame_{frame_count}.png")  # Save each frame as an image
        frame_count += 1

    frame_count += 1  # Increment frame count for the next iteration
    clock.tick(FPS)  # Control the frame rate


# ffmpeg -framerate 20 -i frame_%d.png -c:v libx264 -pix_fmt yuv420p output.mp4