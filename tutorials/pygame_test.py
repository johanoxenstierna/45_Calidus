

import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up display
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Move PNG Example")

# Load the image
image = pygame.image.load('./pictures/Calidus1/planets/6_JupiterL.png')
image_rect = image.get_rect()  # Get the rectangle of the image for positioning

# Set initial position and speed
image_rect.topleft = (100, 100)
speed = [0.5, 0.5]  # Speed in x and y directions

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Move the image
    image_rect.x += speed[0]
    image_rect.y += speed[1]

    # Bounce off the edges
    if image_rect.left < 0 or image_rect.right > width:
        speed[0] = -speed[0]
    if image_rect.top < 0 or image_rect.bottom > height:
        speed[1] = -speed[1]

    # Clear the screen
    screen.fill((255, 255, 255))  # Fill with white
    # Draw the image
    screen.blit(image, image_rect)

    # Update the display
    pygame.display.flip()
    pygame.time.delay(30)  # Control the frame rate
