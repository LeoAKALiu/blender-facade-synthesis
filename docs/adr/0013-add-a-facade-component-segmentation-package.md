# Add a facade-component segmentation package

The first release will generate a separate semantic-segmentation training
package for visible facade components: wall, window glass, window frame, door,
balcony, floor band, podium/storefront, roof/parapet, and background. Classes
will be assigned from Blender scene objects and material/object passes, so the
package is usable for component segmentation without weakening the independent
window-count, floorline, floor-count, or building-use contracts.

## Consequences

- The vocabulary is a versioned public data contract and must be validated for
  every sample.
- A scene component can exist without appearing in a particular view's visible
  semantic mask.
