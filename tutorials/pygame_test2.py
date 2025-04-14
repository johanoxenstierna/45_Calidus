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

# Pre-computed transformations (example values)
translations = np.random.rand(5, 2) * 2  # Random translations

# Set initial positions
for i, rect in enumerate(image_rects):
    rect.topleft = (np.random.randint(0, width), np.random.randint(0, height))

# Main loop
WRITE = True  # Set to True to write to file
FPS = 20
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear the screen
    screen.blit(background, (0, 0))

    # Update and draw images
    for i, rect in enumerate(image_rects):
        # Apply transformations
        rect.x += translations[i][0]
        rect.y += translations[i][1]

        # Check if the image is still on the screen
        if rect.left < 0 or rect.right > width or rect.top < 0 or rect.bottom > height:
            # Reset position if it goes off-screen
            rect.topleft = (np.random.randint(0, width), np.random.randint(0, height))

        # Draw the image
        screen.blit(images[i], rect)

    # Update the display
    pygame.display.flip()
    clock.tick(FPS)  # Control the frame rate
