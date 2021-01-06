from georeferencing.georef import georef_segmentation, geojson_segmentation, transform_coordinates, \
    geojson_transform
from geojson import Feature, Polygon, FeatureCollection
import cv2, os, re, json, tqdm
from osgeo import ogr, osr, gdal


class GeoReferencing:
    def __init__(self, orthophoto_path):
        self.cropsize_x = 512
        self.cropsize_y = 512
        self.orthophoto_path = orthophoto_path

        dataset = gdal.Open(orthophoto_path, gdal.GA_ReadOnly)

        self.ul_x, self.pixel_size, b, self.ul_y, d, pixelHeight = dataset.GetGeoTransform()
        self.prj = dataset.GetProjection()
        ortho_height, ortho_width = dataset.RasterYSize, dataset.RasterXSize

        self.num_h = ortho_height // self.cropsize_y if ortho_height % self.cropsize_y == 0 else ortho_height // self.cropsize_y + 1
        self.num_w = ortho_width // self.cropsize_x if ortho_width % self.cropsize_x == 0 else ortho_width // self.cropsize_x + 1

        self.inference_geo = []

    def __call__(self,object_coords, img_path):

        # inference_metadata = []
        for idx,img_dir in tqdm.tqdm(enumerate(img_path)):
            self.name = os.path.split(img_dir)[-1][:-4]
            file_num = int(re.findall("\d+", self.name)[0])
            self.read_col, self.read_row = \
                [[col, row] for row in range(self.num_h) for col in range(self.num_w) if
                 (self.num_w * row) + col == file_num][0]

            inference_metadata = []
            for segmentations_px in object_coords[idx]:
                inference_world = georef_segmentation([self.ul_x, self.ul_y], self.pixel_size, self.read_row,
                                                      self.read_col,
                                                      segmentations_px, self.cropsize_x, self.cropsize_y)  # unit: m, m
                inference_metadata.append(geojson_segmentation(segmentations_px[-1],
                                                               segmentations_px[:-1], inference_world))

            for meta in inference_metadata:
                if meta["obj_boundary_world"] is False:
                    continue
                geo_data = Feature(geometry=meta["obj_boundary_world"],
                                   properties={
                                       "name": meta["class"],#"inference",
                                       "class": meta["class"],
                                       # "canvas": meta["canvas"]
                                   })
                self.inference_geo.append(geo_data)
                geo_json_result = FeatureCollection([geo_data, *self.inference_geo],
                                                    crs={"type": "name", "properties": {"name": "EPSG:5186"}})
                geojson_transform(geo_json_result, os.path.split(self.orthophoto_path)[-1][:-4], self.prj, 5186)

        return inference_metadata











