"""
Orthoprojection module for GNSS-denied visual localization.

Transforms a downward-facing camera image into a top-down (nadir) view
at a fixed resolution (1 m/pixel), using altitude from an altimeter and
roll/pitch from an IMU.

Reference:
    Kinnari et al., "Season-Invariant GNSS-Denied Visual Localization for
    UAVs," IEEE RA-L, 2022, Section IV-C:
    "orthoprojection can be done ... by the use of calibrated downward-
    facing camera and an altimeter."

    Full VIO-based method in:
    Kinnari et al., "GNSS-denied geolocalization of UAVs by visual matching
    of onboard camera images with orthophotos," ICAR 2021, Section III-C.
"""

import numpy as np
import cv2


def ground_frame(roll, pitch):
    """Ground-plane frame in camera coordinates for a tilted down-facing camera.

    Returns (plane_normal, x_axis, y_axis):
      plane_normal: unit normal of the ground plane in the camera frame,
                    pointing from the camera toward the ground.
      x_axis:       ground-axis pointing right in the camera image.
      y_axis:       ground-axis pointing down in the camera image.

    With zero tilt, x_axis = camera x and y_axis = camera y, so the
    orthoprojection preserves the camera image orientation.
    """
    # Camera tilt rotation (roll around x, pitch around y)
    R_roll = np.array([
        [1, 0,            0            ],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)],
    ])
    R_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0,              1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    R_tilt = R_pitch @ R_roll

    # In nadir orientation, gravity aligns with camera z: [0, 0, 1].
    # After tilting the camera by R_tilt, gravity in camera frame becomes
    # R_tilt^T @ [0, 0, 1]. The ground plane is at distance = altitude.
    gravity_cam = R_tilt.T @ np.array([0.0, 0.0, 1.0])
    plane_normal = gravity_cam / np.linalg.norm(gravity_cam)

    cam_x = np.array([1.0, 0.0, 0.0])
    x_axis = cam_x - np.dot(cam_x, plane_normal) * plane_normal
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(plane_normal, x_axis)
    return plane_normal, x_axis, y_axis


def project_corners_to_ground(W, H, altitude, roll, pitch, fx, fy, cx, cy):
    """Backproject the 4 image corners onto the ground plane.

    Returns (corners_px, ground_2d):
      corners_px: (4,2) float — UL, UR, LR, LL pixel coordinates
      ground_2d:  (4,2) float — camera-relative ground coordinates (x=right,
                  y=down-in-image) in meters. The nadir point (directly below
                  the camera) is ground (0, 0).
    """
    corners_px = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H],
    ], dtype=np.float32)

    # Ray direction for pixel (u, v): d = [(u - cx)/fx, (v - cy)/fy, 1.0]
    rays = np.column_stack([
        (corners_px[:, 0] - cx) / fx,
        (corners_px[:, 1] - cy) / fy,
        np.ones(4),
    ])

    plane_normal, x_axis, y_axis = ground_frame(roll, pitch)

    # Ray-plane intersection: P = t * d with n . P = altitude => t = altitude/(n.d)
    ground_2d = np.empty((4, 2))
    for i in range(4):
        denom = np.dot(plane_normal, rays[i])
        if abs(denom) < 1e-6:
            # Ray nearly parallel to ground — clamp to a far distance
            P = rays[i] * 1e6
        else:
            P = rays[i] * (altitude / denom)
        ground_2d[i] = (np.dot(P, x_axis), np.dot(P, y_axis))
    return corners_px, ground_2d


def orthoproject(image, altitude, roll, pitch, fx, fy, cx, cy, resolution=1.0):
    """
    Orthoproject a downward-facing camera image to a top-down view.

    Camera convention (OpenCV pinhole): x=right, y=down, z=forward.
    The camera is nominally pointing downward, with tilt described by
    roll (around x) and pitch (around y) from the IMU. The ground is
    assumed planar at the given altitude below the camera.

    Args:
        image:      H x W x 3 uint8 array (camera image)
        altitude:   height above ground in meters (from altimeter)
        roll:       camera roll in radians (from IMU, around camera x-axis)
        pitch:      camera pitch in radians (from IMU, around camera y-axis)
        fx, fy:     focal length in pixels (from camera calibration)
        cx, cy:     principal point in pixels (from camera calibration)
        resolution: output resolution in m/pixel (default 1.0)

    Returns:
        orthoprojection: H_out x W_out x 3 uint8 array (top-down view)
        mask:            H_out x W_out uint8 array (255=valid, 0=invalid)
        extent:          (x_min, x_max, y_min, y_max) ground coverage in meters
                         in camera-relative ground coords (x=right, y=down in
                         image); the nadir point is ground (0, 0)
    """
    H_img, W_img = image.shape[:2]

    corners_px, ground_2d = project_corners_to_ground(
        W_img, H_img, altitude, roll, pitch, fx, fy, cx, cy)

    # Output bounds and size
    x_min, x_max = ground_2d[:, 0].min(), ground_2d[:, 0].max()
    y_min, y_max = ground_2d[:, 1].min(), ground_2d[:, 1].max()
    out_w = max(1, int(np.ceil((x_max - x_min) / resolution)))
    out_h = max(1, int(np.ceil((y_max - y_min) / resolution)))

    # Map ground coordinates to output pixel coordinates
    dst_corners = np.zeros((4, 2), dtype=np.float32)
    dst_corners[:, 0] = (ground_2d[:, 0] - x_min) / resolution
    dst_corners[:, 1] = (ground_2d[:, 1] - y_min) / resolution

    # Homography + warp
    H, _ = cv2.findHomography(corners_px, dst_corners)
    orthoprojection = cv2.warpPerspective(image, H, (out_w, out_h))
    mask = cv2.warpPerspective(
        np.ones((H_img, W_img), dtype=np.uint8) * 255, H, (out_w, out_h)
    )

    extent = (float(x_min), float(x_max), float(y_min), float(y_max))
    return orthoprojection, mask, extent


def extract_patch(orthoprojection, mask, extent, size_m=96.0, resolution=1.0):
    """
    Extract a square patch from the orthoprojection, centered on the nadir.

    The patch is centered on the camera's nadir point (ground (0,0) =
    pixel (-x_min/res, -y_min/res)) — NOT the output image center, which
    differs when the camera is tilted. If the nadir-centered window is
    not fully inside the valid footprint, it is shifted to the nearest
    fully-valid position (paper Fig. 9(a): "adjust position to ensure
    full visibility in the camera image"), avoiding zero-padding.

    Args:
        orthoprojection: H x W x 3 array from orthoproject()
        mask:           H x W array from orthoproject()
        extent:         (x_min, x_max, y_min, y_max) from orthoproject()
        size_m:         patch size in meters (default 96, matching the paper)
        resolution:     m/pixel of the orthoprojection (default 1.0)

    Returns:
        patch:        size_px x size_px x 3 array (the cropped patch)
        patch_mask:   size_px x size_px array
        center_ground: (x, y) ground coords of the patch CENTER in meters,
                      camera-relative (x=right, y=down in image). The nadir
                      is (0, 0); a non-zero value means the patch was shifted.
    """
    size_px = int(round(size_m / resolution))
    h, w = orthoprojection.shape[:2]
    x_min, _, y_min, _ = extent

    # Nadir point in output pixels (ground (0,0))
    nadir_px = int(round(-x_min / resolution))
    nadir_py = int(round(-y_min / resolution))

    x0 = nadir_px - size_px // 2
    y0 = nadir_py - size_px // 2

    # Shift the window to the nearest position where it is fully valid
    if mask is not None:
        valid = cv2.erode(mask, np.ones((size_px, size_px), np.uint8))
        ctr_x, ctr_y = x0 + size_px // 2, y0 + size_px // 2
        ok = (0 <= ctr_y < h and 0 <= ctr_x < w and valid[ctr_y, ctr_x] > 0)
        if not ok:
            ys, xs = np.nonzero(valid)
            if len(ys) > 0:
                d2 = (ys - ctr_y) ** 2 + (xs - ctr_x) ** 2
                i = int(np.argmin(d2))
                ctr_y, ctr_x = int(ys[i]), int(xs[i])
                x0 = ctr_x - size_px // 2
                y0 = ctr_y - size_px // 2

    # Pad with zeros only if the window still extends beyond the image
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, (y0 + size_px) - h)
    pad_right = max(0, (x0 + size_px) - w)

    if pad_top or pad_left or pad_bottom or pad_right:
        orthoprojection = cv2.copyMakeBorder(
            orthoprojection, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        mask = cv2.copyMakeBorder(
            mask, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        x0 += pad_left
        y0 += pad_top

    patch = orthoprojection[y0:y0 + size_px, x0:x0 + size_px]
    patch_mask = mask[y0:y0 + size_px, x0:x0 + size_px]

    # Ground coords of the actual patch center
    center_ground = (
        (x0 + size_px / 2.0) * resolution + x_min,
        (y0 + size_px / 2.0) * resolution + y_min,
    )
    return patch, patch_mask, center_ground


if __name__ == "__main__":
    # --- Demo with a synthetic checkerboard image ---
    # Simulates a downward-facing camera viewing a checkerboard pattern on
    # the ground. With zero tilt, the orthoprojection should look identical
    # to the original (just scaled to 1 m/px). With tilt, it should show
    # the perspective correction.

    import os

    # Create a synthetic camera image (640x512, checkerboard pattern)
    img_h, img_w = 512, 640
    square = 40  # pixels per checker square
    # image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # for i in range(0, img_h, square):
    #     for j in range(0, img_w, square):
    #         if ((i // square) + (j // square)) % 2 == 0:
    #             image[i:i+square, j:j+square] = [200, 200, 200]
    #         else:
    #             image[i:i+square, j:j+square] = [60, 60, 60]

    image = cv2.imread("sivl/data/google_earth_exports/area1/2018_5_1.jpg")
    img_h, img_w =image.shape[:2]

    # Add a marker at the center (crosshair)
    cv2.drawMarker(image, (img_w//2, img_h//2), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)

    # Camera parameters (typical for a downward-facing camera)
    fx, fy = 200.0, 200.0       # focal length in pixels
    cx, cy = img_w / 2, img_h / 2  # principal point at image center
    altitude = 100.0            # meters above ground
    resolution = 1.0            # m/pixel output

    # Test 1: No tilt (should produce an undistorted top-down view)
    print("=== Test 1: No tilt (roll=0, pitch=0) ===")
    ortho, mask_img, extent = orthoproject(
        image, altitude, roll=0.0, pitch=0.0,
        fx=fx, fy=fy, cx=cx, cy=cy, resolution=resolution
    )
    print(f"  Input image:     {img_w} x {img_h} px")
    print(f"  Altitude:        {altitude} m")
    print(f"  Ground extent:   {extent[1]-extent[0]:.1f} x {extent[3]-extent[2]:.1f} m")
    print(f"  Orthoprojection: {ortho.shape[1]} x {ortho.shape[0]} px")
    print(f"  Extent (m):      x=[{extent[0]:.1f}, {extent[1]:.1f}], "
          f"y=[{extent[2]:.1f}, {extent[3]:.1f}]")

    patch, patch_mask, center_ground = extract_patch(
        ortho, mask_img, extent, size_m=96, resolution=resolution)
    print(f"  96m patch:       {patch.shape[1]} x {patch.shape[0]} px  "
          f"center={center_ground}")

    # Test 2: With tilt (roll=10deg, pitch=5deg)
    print("\n=== Test 2: With tilt (roll=10deg, pitch=5deg) ===")
    ortho_tilt, mask_tilt, extent_tilt = orthoproject(
        image, altitude,
        roll=np.radians(10), pitch=np.radians(5),
        fx=fx, fy=fy, cx=cx, cy=cy, resolution=resolution
    )
    print(f"  Orthoprojection: {ortho_tilt.shape[1]} x {ortho_tilt.shape[0]} px")
    print(f"  Extent (m):      x=[{extent_tilt[0]:.1f}, {extent_tilt[1]:.1f}], "
          f"y=[{extent_tilt[2]:.1f}, {extent_tilt[3]:.1f}]")

    # Save results for visual inspection
    out_dir = os.path.join(os.path.dirname(__file__), "sample_images", "orthoprojection_demo")
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "input.png"), image)
    cv2.imwrite(os.path.join(out_dir, "ortho_nadir.png"), ortho)
    cv2.imwrite(os.path.join(out_dir, "ortho_tilted.png"), ortho_tilt)
    cv2.imwrite(os.path.join(out_dir, "mask_nadir.png"), mask_img)
    cv2.imwrite(os.path.join(out_dir, "patch_nadir.png"), patch)

    print(f"\n  Results saved to: {out_dir}")
    print("  - input.png        : original camera image")
    print("  - ortho_nadir.png  : orthoprojection (no tilt)")
    print("  - ortho_tilted.png : orthoprojection (with tilt)")
    print("  - mask_nadir.png   : validity mask")
    print("  - patch_nadir.png  : 96x96m centered patch")
