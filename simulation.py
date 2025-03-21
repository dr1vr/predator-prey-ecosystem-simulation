#!/usr/bin/env python3
"""
Simple Predator-Prey Ecosystem Simulation - Alpha Version
This is a simplified version that demonstrates the basic concepts.
"""

import pygame
import random
import math
import numpy as np
from typing import List, Tuple, Dict, Any

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
PLANT_COLOR = (0, 100, 0)  # Dark green
HERBIVORE_COLOR = (0, 0, 200)  # Blue
CARNIVORE_COLOR = (200, 0, 0)  # Red
WATER_COLOR = (0, 50, 200)  # Blue-ish
BACKGROUND_COLOR = (30, 30, 50)  # Dark blue-grey
FPS = 60

class Entity:
    """Base class for all entities in the simulation."""
    
    def __init__(self, x: float, y: float, size: float = 5.0):
        """Initialize an entity."""
        self.x = x
        self.y = y
        self.size = size
        self.color = (255, 255, 255)  # Default: white
        self.alive = True
        self.age = 0
        self.max_age = 100
        self.energy = 100
        self.max_energy = 100
        self.velocity = [0, 0]
    
    def update(self, dt: float, simulation):
        """Update entity state."""
        # Update age
        self.age += dt
        if self.age > self.max_age:
            self.alive = False
            
        # Basic physics - apply velocity
        self.x += self.velocity[0] * dt * 10
        self.y += self.velocity[1] * dt * 10
        
        # Apply friction
        self.velocity[0] *= 0.95
        self.velocity[1] *= 0.95
        
        # Boundary checking
        self.x = max(0, min(SCREEN_WIDTH, self.x))
        self.y = max(0, min(SCREEN_HEIGHT, self.y))
        
        # Energy consumption
        self.energy -= 0.5 * dt
        if self.energy <= 0:
            self.alive = False
    
    def render(self, screen):
        """Render the entity on screen."""
        if self.alive:
            pygame.draw.circle(
                screen, 
                self.color, 
                (int(self.x), int(self.y)), 
                int(self.size)
            )
    
    def distance_to(self, other) -> float:
        """Calculate distance to another entity."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def is_alive(self) -> bool:
        """Check if entity is alive."""
        return self.alive

class Plant(Entity):
    """Plant entity that grows and can be eaten by herbivores."""
    
    def __init__(self, x: float, y: float):
        """Initialize a plant."""
        super().__init__(x, y, size=3.0)
        self.color = PLANT_COLOR
        self.max_size = 8.0
        self.growth_rate = 0.2
        self.energy_value = 30
        self.max_age = 200
        self.reproduction_chance = 0.001
    
    def update(self, dt: float, simulation):
        """Update plant state."""
        super().update(dt, simulation)
        
        # Growth
        if self.size < self.max_size:
            self.size += self.growth_rate * dt
            
        # Reproduction
        if random.random() < self.reproduction_chance * dt and self.size > self.max_size * 0.7:
            # Try to reproduce
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(10, 30)
            new_x = self.x + math.cos(angle) * distance
            new_y = self.y + math.sin(angle) * distance
            
            # Check bounds
            if 0 <= new_x < SCREEN_WIDTH and 0 <= new_y < SCREEN_HEIGHT:
                simulation.add_entity(Plant(new_x, new_y))
                self.size *= 0.8  # Reduce size after reproducing

class Herbivore(Entity):
    """Herbivore that eats plants and can be eaten by carnivores."""
    
    def __init__(self, x: float, y: float):
        """Initialize a herbivore."""
        super().__init__(x, y, size=6.0)
        self.color = HERBIVORE_COLOR
        self.speed = 40.0
        self.perception_range = 100.0
        self.max_energy = 150
        self.energy = self.max_energy * 0.8
        self.hunger = 0.5
        self.energy_consumption_rate = 5.0
        self.reproduction_chance = 0.0005
        self.max_age = 150
    
    def update(self, dt: float, simulation):
        """Update herbivore state."""
        super().update(dt, simulation)
        
        # Increase hunger
        self.hunger = min(1.0, self.hunger + 0.02 * dt)
        
        # Find food if hungry
        if self.hunger > 0.5:
            self.find_and_eat_plants(simulation)
        
        # Move randomly if no specific goal
        if random.random() < 0.05:
            angle = random.uniform(0, 2 * math.pi)
            self.velocity[0] = math.cos(angle) * self.speed * 0.1
            self.velocity[1] = math.sin(angle) * self.speed * 0.1
            
        # Try to reproduce
        if (random.random() < self.reproduction_chance * dt and 
                self.energy > self.max_energy * 0.7):
            self.reproduce(simulation)
    
    def find_and_eat_plants(self, simulation):
        """Find and eat nearby plants."""
        for entity in simulation.entities:
            if (isinstance(entity, Plant) and entity.is_alive() and 
                    self.distance_to(entity) < self.perception_range):
                # Move toward plant
                direction_x = entity.x - self.x
                direction_y = entity.y - self.y
                distance = max(0.1, math.sqrt(direction_x**2 + direction_y**2))
                
                # Normalize and apply speed
                self.velocity[0] = direction_x / distance * self.speed * 0.1
                self.velocity[1] = direction_y / distance * self.speed * 0.1
                
                # If close enough, eat plant
                if self.distance_to(entity) < self.size + entity.size:
                    self.energy = min(self.max_energy, self.energy + entity.energy_value)
                    self.hunger = max(0, self.hunger - 0.3)
                    entity.alive = False
                    break
    
    def reproduce(self, simulation):
        """Create a new herbivore nearby."""
        angle = random.uniform(0, 2 * math.pi)
        distance = self.size * 2
        new_x = self.x + math.cos(angle) * distance
        new_y = self.y + math.sin(angle) * distance
        
        # Check bounds
        if 0 <= new_x < SCREEN_WIDTH and 0 <= new_y < SCREEN_HEIGHT:
            simulation.add_entity(Herbivore(new_x, new_y))
            self.energy *= 0.7  # Reduce energy after reproducing

class Carnivore(Entity):
    """Carnivore that hunts herbivores."""
    
    def __init__(self, x: float, y: float):
        """Initialize a carnivore."""
        super().__init__(x, y, size=8.0)
        self.color = CARNIVORE_COLOR
        self.speed = 60.0
        self.perception_range = 150.0
        self.max_energy = 200
        self.energy = self.max_energy * 0.8
        self.hunger = 0.5
        self.energy_consumption_rate = 8.0
        self.reproduction_chance = 0.0002
        self.max_age = 180
    
    def update(self, dt: float, simulation):
        """Update carnivore state."""
        super().update(dt, simulation)
        
        # Increase hunger
        self.hunger = min(1.0, self.hunger + 0.015 * dt)
        
        # Find prey if hungry
        if self.hunger > 0.4:
            self.hunt_herbivores(simulation)
        
        # Move randomly if no specific goal
        if random.random() < 0.03:
            angle = random.uniform(0, 2 * math.pi)
            self.velocity[0] = math.cos(angle) * self.speed * 0.1
            self.velocity[1] = math.sin(angle) * self.speed * 0.1
            
        # Try to reproduce
        if (random.random() < self.reproduction_chance * dt and 
                self.energy > self.max_energy * 0.8):
            self.reproduce(simulation)
    
    def hunt_herbivores(self, simulation):
        """Find and hunt nearby herbivores."""
        for entity in simulation.entities:
            if (isinstance(entity, Herbivore) and entity.is_alive() and 
                    self.distance_to(entity) < self.perception_range):
                # Move toward herbivore
                direction_x = entity.x - self.x
                direction_y = entity.y - self.y
                distance = max(0.1, math.sqrt(direction_x**2 + direction_y**2))
                
                # Normalize and apply speed
                self.velocity[0] = direction_x / distance * self.speed * 0.1
                self.velocity[1] = direction_y / distance * self.speed * 0.1
                
                # If close enough, eat herbivore
                if self.distance_to(entity) < self.size + entity.size:
                    self.energy = min(self.max_energy, self.energy + entity.energy)
                    self.hunger = max(0, self.hunger - 0.5)
                    entity.alive = False
                    break
    
    def reproduce(self, simulation):
        """Create a new carnivore nearby."""
        angle = random.uniform(0, 2 * math.pi)
        distance = self.size * 2
        new_x = self.x + math.cos(angle) * distance
        new_y = self.y + math.sin(angle) * distance
        
        # Check bounds
        if 0 <= new_x < SCREEN_WIDTH and 0 <= new_y < SCREEN_HEIGHT:
            simulation.add_entity(Carnivore(new_x, new_y))
            self.energy *= 0.6  # Reduce energy after reproducing

class Environment:
    """Simple environment with water bodies."""
    
    def __init__(self, width: int, height: int):
        """Initialize the environment."""
        self.width = width
        self.height = height
        self.water_grid = np.zeros((width, height))
        
        # Create some water bodies
        self.create_water_bodies()
    
    def create_water_bodies(self):
        """Create water bodies in the environment."""
        # Create a lake
        lake_x = self.width // 3
        lake_y = self.height // 2
        lake_radius = min(self.width, self.height) // 6
        
        for x in range(max(0, lake_x - lake_radius), min(self.width, lake_x + lake_radius)):
            for y in range(max(0, lake_y - lake_radius), min(self.height, lake_y + lake_radius)):
                dist = math.sqrt((x - lake_x) ** 2 + (y - lake_y) ** 2)
                if dist < lake_radius:
                    # Use gaussian distribution for natural look
                    water_level = math.exp(-(dist/lake_radius) ** 2)
                    self.water_grid[x, y] = water_level
        
        # Create a river
        start_x = 0
        start_y = self.height // 3
        end_x = self.width
        end_y = self.height // 3 * 2
        
        steps = self.width // 2
        for t in range(steps):
            t_norm = t / steps
            # Cubic Bezier for curved river
            x = int(start_x + t_norm * (end_x - start_x))
            y = int(start_y + math.sin(t_norm * math.pi) * (end_y - start_y))
            
            river_width = 10 + 5 * math.sin(t_norm * math.pi * 2)
            
            for dx in range(-int(river_width), int(river_width)):
                for dy in range(-int(river_width), int(river_width)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        dist = math.sqrt(dx**2 + dy**2)
                        if dist < river_width:
                            water_level = math.exp(-(dist/river_width) ** 2)
                            self.water_grid[nx, ny] = max(self.water_grid[nx, ny], water_level)
    
    def render(self, screen):
        """Render the environment on screen."""
        # Render water bodies
        water_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for x in range(0, self.width, 4):  # Sample every 4 pixels for performance
            for y in range(0, self.height, 4):
                water_level = self.water_grid[x, y]
                if water_level > 0.1:
                    # Calculate water color based on depth
                    alpha = int(100 * water_level)
                    water_color = (*WATER_COLOR, alpha)
                    
                    pygame.draw.rect(
                        water_surface, 
                        water_color, 
                        (x, y, 4, 4)
                    )
        
        screen.blit(water_surface, (0, 0))

class Simulation:
    """Main simulation class that manages entities and updates."""
    
    def __init__(self):
        """Initialize the simulation."""
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Predator-Prey Ecosystem Simulation - Alpha")
        
        self.clock = pygame.time.Clock()
        self.entities = []
        self.environment = Environment(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.running = True
        self.paused = False
        self.time_scale = 1.0
        self.day_night_cycle = 0.0  # 0.0 to 1.0 (day/night cycle)
        
        # Stats
        self.stats = {
            "plants": 0,
            "herbivores": 0,
            "carnivores": 0
        }
        
        # Initialize font for stats display
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Initialize population history for graph
        self.population_history = {
            "plants": [],
            "herbivores": [],
            "carnivores": []
        }
        self.show_graph = False
        
        # Initialize with some entities
        self.populate_initial_entities()
    
    def populate_initial_entities(self):
        """Create initial entities."""
        # Add plants
        for _ in range(100):
            x = random.uniform(0, SCREEN_WIDTH)
            y = random.uniform(0, SCREEN_HEIGHT)
            self.add_entity(Plant(x, y))
        
        # Add herbivores
        for _ in range(20):
            x = random.uniform(0, SCREEN_WIDTH)
            y = random.uniform(0, SCREEN_HEIGHT)
            self.add_entity(Herbivore(x, y))
        
        # Add carnivores
        for _ in range(5):
            x = random.uniform(0, SCREEN_WIDTH)
            y = random.uniform(0, SCREEN_HEIGHT)
            self.add_entity(Carnivore(x, y))
    
    def add_entity(self, entity):
        """Add an entity to the simulation."""
        self.entities.append(entity)
    
    def update(self, dt):
        """Update all entities in the simulation."""
        # Skip if paused
        if self.paused:
            return
            
        # Apply time scaling
        dt *= self.time_scale
        
        # Update day/night cycle (complete cycle every 60 seconds)
        self.day_night_cycle = (self.day_night_cycle + dt / 60.0) % 1.0
        
        # Update entities
        for entity in list(self.entities):
            if entity.is_alive():
                entity.update(dt, self)
        
        # Clean up dead entities
        self.entities = [e for e in self.entities if e.is_alive()]
        
        # Update stats
        self.update_stats()
    
    def update_stats(self):
        """Update simulation statistics."""
        plants = sum(1 for e in self.entities if isinstance(e, Plant))
        herbivores = sum(1 for e in self.entities if isinstance(e, Herbivore))
        carnivores = sum(1 for e in self.entities if isinstance(e, Carnivore))
        
        self.stats = {
            "plants": plants,
            "herbivores": herbivores,
            "carnivores": carnivores
        }
        
        # Add to history (limit history length to 200 points)
        for key in self.population_history:
            self.population_history[key].append(self.stats.get(key, 0))
            if len(self.population_history[key]) > 200:
                self.population_history[key].pop(0)
    
    def render(self):
        """Render all entities and UI elements."""
        # Calculate background color based on day/night cycle
        day_factor = math.sin(self.day_night_cycle * math.pi)
        bg_color = [
            int(max(5, BACKGROUND_COLOR[0] * day_factor * 2)), 
            int(max(5, BACKGROUND_COLOR[1] * day_factor * 2)), 
            int(max(10, BACKGROUND_COLOR[2] * day_factor * 1.5))
        ]
        
        # Fill the screen with background color
        self.screen.fill(bg_color)
        
        # Render environment
        self.environment.render(self.screen)
        
        # Render all entities
        for entity in self.entities:
            entity.render(self.screen)
        
        # Render stats text
        stats_text = f"Plants: {self.stats['plants']}  Herbivores: {self.stats['herbivores']}  Carnivores: {self.stats['carnivores']}  Time Scale: {self.time_scale:.1f}x"
        stats_surface = self.font.render(stats_text, True, (255, 255, 255))
        self.screen.blit(stats_surface, (10, 10))
        
        # Render status (paused/running)
        status_text = "PAUSED" if self.paused else "RUNNING"
        status_surface = self.font.render(status_text, True, (255, 255, 0) if self.paused else (0, 255, 0))
        self.screen.blit(status_surface, (SCREEN_WIDTH - 100, 10))
        
        # Render population graph if enabled
        if self.show_graph:
            self.render_population_graph()
    
    def render_population_graph(self):
        """Render population history graph."""
        graph_height = 150
        graph_width = 400
        graph_x = SCREEN_WIDTH - graph_width - 10
        graph_y = SCREEN_HEIGHT - graph_height - 10
        
        # Draw graph background
        graph_surface = pygame.Surface((graph_width, graph_height))
        graph_surface.set_alpha(180)
        graph_surface.fill((0, 0, 0))
        
        # Draw graph lines
        colors = {
            "plants": PLANT_COLOR,
            "herbivores": HERBIVORE_COLOR,
            "carnivores": CARNIVORE_COLOR
        }
        
        for species, history in self.population_history.items():
            if not history:
                continue
            
            # Scale to fit graph
            max_val = max(max(h) for h in self.population_history.values() if h) or 1
            scale_y = (graph_height - 20) / max_val
            
            # Draw line
            points = []
            for i, val in enumerate(history):
                x = graph_x + 10 + i * (graph_width - 20) / (len(history) - 1 if len(history) > 1 else 1)
                y = graph_y + graph_height - 10 - val * scale_y
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, colors[species], False, points, 2)
        
        # Draw graph border
        pygame.draw.rect(self.screen, (150, 150, 150), 
                        (graph_x, graph_y, graph_width, graph_height), 1)
        
        # Draw legend
        legend_y = graph_y + 10
        for species, color in colors.items():
            legend_text = f"{species.capitalize()}: {self.stats[species]}"
            legend_surface = self.font.render(legend_text, True, color)
            self.screen.blit(legend_surface, (graph_x + 15, legend_y))
            legend_y += 20
    
    def run(self):
        """Run the main simulation loop."""
        last_time = pygame.time.get_ticks() / 1000.0
        
        while self.running:
            # Calculate delta time
            current_time = pygame.time.get_ticks() / 1000.0
            dt = current_time - last_time
            last_time = current_time
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.time_scale = min(10.0, self.time_scale * 1.5)
                    elif event.key == pygame.K_DOWN:
                        self.time_scale = max(0.1, self.time_scale / 1.5)
                    elif event.key == pygame.K_g:
                        self.show_graph = not self.show_graph
            
            # Update simulation
            self.update(dt)
            
            # Render
            self.render()
            
            # Update display
            pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run() 