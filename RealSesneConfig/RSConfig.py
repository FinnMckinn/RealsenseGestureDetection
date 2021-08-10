class RealSenseConfig:

    def __init__(self):
        self.pipeline = None
        self.align = None
        self.clipping_distance = None

    def setup(self, rs):
    # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Configure Resolution
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

        # Start streaming
        profile = self.pipeline.start(config)

        # Set Depth Scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Depth Trheshold
        clipping_distance_in_meters = 0.5 # Meters
        self.clipping_distance = clipping_distance_in_meters / depth_scale

        # Align frames
        align_to = rs.stream.color
        self.align = rs.align(align_to)

class RealSenseFilters:

    def __init__(self, rs):
        self.spatial_f = rs.spatial_filter()
        self.temporal_f = rs.temporal_filter()

    def apply_filters(self, aligned_frames):
        depth = aligned_frames.get_depth_frame()
        depth = self.spatial_f.process(depth)
        aligned_depth_frame = self.temporal_f.process(depth)
        return aligned_depth_frame