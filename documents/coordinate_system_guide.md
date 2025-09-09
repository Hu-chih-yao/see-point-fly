# Coordinate System & Projection Guide

This document explains the coordinate system used in the drone project, the mathematics behind 2D-3D projections, and how monitor resolution affects these calculations.

## Coordinate Systems

The drone project uses two primary coordinate systems:

### 1. 3D World Coordinate System

The 3D coordinate system uses a right-handed orientation:
- **X-axis**: Left (-) to Right (+)
- **Y-axis**: Forward distance/depth (+)
- **Z-axis**: Down (-) to Up (+)

The origin (0,0,0) represents the drone's position at any given moment.

```
    Z+  
    ^   
    |   
    |   
    +----> X+
   /    
  /     
 Y+     
```

Typical ranges used in the action space:
```python
self.x_range = (-3.0, 3.0)    # Left/Right range
self.y_range = (0.5, 2.0)     # Forward depth range
self.z_range = (-1.8, 1.8)    # Up/Down range
```

### 2. 2D Screen Coordinate System

The 2D screen coordinate system follows standard image conventions:
- **X-axis**: Left (0) to Right (+width)
- **Y-axis**: Top (0) to Bottom (+height)

The center is located at (width/2, height/2).

```
 (0,0) ---> X+
   |
   |
   v
   Y+
```

## Projection Mathematics

### 3D to 2D Projection

The `project_point` method transforms 3D world coordinates into 2D screen coordinates:

```python
def project_point(self, point_3d: Tuple[float, float, float]) -> Tuple[int, int]:
    """Project 3D point using proper perspective projection for drone view"""
    x, y, z = point_3d
    
    # Center points
    center_x = self.image_width / 2
    center_y = self.image_height / 2
    
    # Calculate perspective scaling based on field of view
    fov_factor = np.tan(np.radians(self.fov_horizontal / 2))
    
    # Perspective projection with proper FOV
    # y is our depth (forward distance)
    if y < 0.1:  # Avoid division by zero
        y = 0.1
        
    # Scale x and z based on perspective and FOV
    x_projected = (x / (y * fov_factor)) * (self.image_width / 2)
    z_projected = (z / (y * fov_factor)) * (self.image_height / 2)
    
    # Calculate final screen coordinates
    screen_x = int(center_x + x_projected)
    screen_y = int(center_y - z_projected)  # Negative because screen y increases downward
    
    return (screen_x, screen_y)
```

This projection follows these key principles:
1. **Perspective Division**: Divide by depth (y-coordinate) for perspective effect
2. **FOV Scaling**: Scale based on field-of-view angle
3. **Center Offset**: Add center offset to place origin at screen center
4. **Y-Inversion**: Flip the z-coordinate because screen Y increases downward

### 2D to 3D Back-Projection

The `reverse_project_point` method performs the inverse operation, estimating a 3D position from a 2D screen coordinate:

```python
def reverse_project_point(self, point_2d: Tuple[int, int], depth: float = 1) -> Tuple[float, float, float]:
    """Project 2D image point back to 3D space"""
    # Center and normalize coordinates
    x_normalized = (point_2d[0] - self.image_width/2) / (self.image_width/2)
    y_normalized = (self.image_height/2 - point_2d[1]) / (self.image_height/2)
    
    # Adjust depth based on vertical position
    depth_factor = 1.0 + (y_normalized * 0.5)
    depth = depth * depth_factor
    
    # Calculate 3D coordinates with optimized depth
    x = depth * x_normalized * np.tan(np.radians(self.fov_horizontal/2))
    z = depth * y_normalized * np.tan(np.radians(self.fov_vertical/2))
    y = depth
    
    return (x, y, z)
```

This back-projection:
1. **Normalizes** the 2D coordinates relative to screen center
2. **Estimates depth** using provided depth value with adjustments
3. **Scales coordinates** based on FOV to match the original projection

## Impact of Monitor Resolution

### Resolution Dependency

The projection formulas rely heavily on accurate screen dimensions:

```python
# Center points
center_x = self.image_width / 2
center_y = self.image_height / 2

# Scale factors
x_projected = (x / (y * fov_factor)) * (self.image_width / 2)
```

When the `image_width` and `image_height` values don't match the actual dimensions of captured images, the projection calculations produce incorrect results.

### Retina/HiDPI Scaling Issue

The resolution mismatch with Retina/HiDPI displays causes:

1. **Incorrect Center Calculation**
   - With `image_width = 1710` but actual width of 3420
   - Center is calculated as 855 instead of 1710
   - All projected points are shifted

2. **Wrong Scaling Factors**
   - Scaling factors are halved relative to what they should be
   - Points appear compressed instead of properly scaled

3. **Visualization Misalignment**
   - All visualizations are drawn at incorrect positions
   - Center point (0,1,0) doesn't project to image center

### Fixed Calculation with Correct Resolution

After updating to use the actual captured resolution, the calculations work correctly:

```python
# Updated in ActionProjector.__init__
self.image_width = 3420
self.image_height = 2214
```

Now the projection works as expected:
- Center point (0,1,0) projects to (1710, 1107)
- All projected points align with visual expectations
- Back-projection accurately reverses the process

## Testing Projection Accuracy

To verify projection accuracy, we use test points with known expected outcomes:

```python
test_points = [
    (0.0, 1.0, 0.0),  # Should project to center
    (1.0, 1.0, 0.0),  # Right of center
    (-1.0, 1.0, 0.0), # Left of center
    (0.0, 1.0, 1.0),  # Above center
    (0.0, 1.0, -1.0)  # Below center
]
```

### Expected Projections with 3420×2214 Resolution

| 3D Point      | Expected 2D Point | Actual Projection |
|---------------|-------------------|-------------------|
| (0.0, 1.0, 0.0) | (1710, 1107)    | (1710, 1107)     |
| (1.0, 1.0, 0.0) | (2952, 1107)    | (2952, 1107)     |
| (-1.0, 1.0, 0.0)| (467, 1107)     | (467, 1107)      |
| (0.0, 1.0, 1.0) | (1710, 302)     | (1710, 302)      |
| (0.0, 1.0, -1.0)| (1710, 1911)    | (1710, 1911)     |

## Debugging Projection Issues

When projection issues occur, follow these steps:

1. Check monitor dimensions:
   ```bash
   python tools/resolution/check_monitors.py
   ```

2. Verify dimensions match in `ActionProjector.__init__`:
   ```python
   # Should match actual screen capture dimensions
   self.image_width = 3420    
   self.image_height = 2214
   ```

3. Test projection pipeline:
   ```bash
   python tools/capture/check_encoding.py
   ```

4. Visualize the coordinate system:
   ```bash
   spf sim --debug
   ```

## Performance Considerations

The projection calculations are relatively lightweight, but when applied to many points they can impact performance:

1. **Forward Projection**:
   - Used when visualizing 3D points on screen
   - Critical for visual feedback
   - Called frequently during visualization

2. **Back Projection**:
   - Used when converting screen points to 3D
   - Critical for interpreting Gemini outputs
   - Less frequently called than forward projection

Optimization tips:
- Cache projection results for frequently used points
- Consider batch processing multiple points
- Avoid unnecessary projections in tight loops
- Use NumPy vectorization for multiple projections 