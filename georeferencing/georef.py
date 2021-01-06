import json
import numpy as np
from logger import logger
# geojson
from geojson import Polygon, Feature
import shapely.wkt
import os

from osgeo import ogr, osr, gdal
from osgeo.osr import SpatialReference, CoordinateTransformation

from rdp import rdp

def transform_coordinates(ul_coord, src_wkt, dst_epsg=3857):
    # Define the source Coordinate System from wkt
    src = SpatialReference(wkt=src_wkt)

    # Define the target coordinate system (EPSG 4326)
    dst = SpatialReference()
    dst.ImportFromEPSG(dst_epsg)

    coord_transformation = CoordinateTransformation(src, dst)

    # Check the transformation for a point close to the centre of the projected grid
    xy = coord_transformation.TransformPoint(float(ul_coord[0]), float(ul_coord[1]))  # The order: X, Y

    return xy[0], xy[1]

def georef_segmentation(ul_coord, ps, r, c, seg_result, inference_size_x, inference_size_y):
    # (10, 10)------------(20:x, 10:y) ---ul_coord(x, y)
    #     |                    |
    #     |                    |
    #    100                   |  (70,30) - seg_result(row, col)
    #     |                    |
    #     | -------100-------- | --------------
    #     |                    |
    #     |                    |
    #     |                    |

    obj_world = np.zeros_like(seg_result[:-1], dtype=float)
    seg_result = np.array(seg_result[:-1], dtype=float)
    obj_world[0::2] = ul_coord[0] + (c * inference_size_x + seg_result[0::2]) * ps  # X_world
    obj_world[1::2] = ul_coord[1] - (r * inference_size_y + seg_result[1::2]) * ps  # Y_world

    return obj_world


def geojson_segmentation(object_type, boundary_image, boundary_world):
    obj_metadata = {
        "class": object_type,
        "canvas": boundary_image
    }

    boundary_list = boundary_world
    boundary_len = len(boundary_list) / 2
    object_boundary = []
    for i in range(int(boundary_len)):
        j = i * 2
        object_boundary.append((boundary_list[j], boundary_list[j + 1]))
    object_boundary.append((boundary_list[0], boundary_list[1]))

    object_rdp = rdp(object_boundary, epsilon=0.5)

    object_list = []
    object_list.append(object_rdp)
    object_boundary_geo = Polygon(object_list)
    if object_boundary_geo.is_valid == False:
        # logger.info('inference geojson False')
        # print('inference geojson False')
        object_boundary_geo = False
        # raise HTTPException(status_code=500, detail='inference geojson False')
    obj_metadata["obj_boundary_world"] = object_boundary_geo  # string in wkt

    return obj_metadata


def geojson_transform(geojson_metadata,fname, src_wkt, dst_epsg=5186):
    # https://gis.stackexchange.com/questions/224413/ogr-api-get-geojson-from-string
    driver = ogr.GetDriverByName('GeoJSON')

    # Define the source Coordinate System from wkt
    inSpatialRef = SpatialReference(wkt=src_wkt)

    # output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(dst_epsg)

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # get the input layer
    inDataSet = gdal.OpenEx(str(geojson_metadata))
    inLayer = inDataSet.GetLayer()

    # create the output layer
    outputShapefile = "%s.geojson" % fname
    if os.path.exists(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer("test", geom_type=ogr.wkbMultiPolygon, srs=outSpatialRef)

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)
    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(1, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), str(inFeature.GetField(i)))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()

    # s = str(geom.ExportToWkt())
    # g1 = shapely.wkt.loads(s)
    # g2 = Feature(geometry=g1, properties={})
    # res_geo = g2.geometry

    # Save and close the shapefiles
    inDataSet = None
    outDataSet = None

    # return res_geo

    # # https://gis.stackexchange.com/questions/224413/ogr-api-get-geojson-from-string
    # # geojson = '{"crs": {"properties": {"name": "EPSG:5186"}, "type": "name"}, "features": [{"geometry": {"coordinates": [[[185814.96, 83849.64], [185813.96, 83846.6], [185813.96, 83815.12], [185851.68, 83815.12], [185852.96, 83815.8], [185850.68, 83818.44], [185847.6, 83819.4], [185842.04, 83825.0], [185839.76, 83825.16], [185835.32, 83830.12], [185828.72, 83834.92], [185828.32, 83837.76], [185824.72, 83838.92], [185819.16, 83844.68], [185817.36, 83845.16], [185816.16, 83846.52], [185816.16, 83849.12], [185814.96, 83849.64]]], "type": "Polygon"}, "properties": {"canvas": [25, 1184, 24, 1185, 24, 1186, 22, 1188, 17, 1188, 16, 1189, 16, 1194, 14, 1196, 13, 1196, 12, 1197, 12, 1202, 10, 1204, 9, 1204, 8, 1205, 8, 1206, 7, 1207, 6, 1207, 4, 1209, 4, 1210, 6, 1212, 7, 1212, 8, 1213, 8, 1238, 6, 1240, 5, 1240, 4, 1241, 4, 1258, 2, 1260, 0, 1260, 0, 2047, 943, 2047, 943, 2045, 945, 2043, 950, 2043, 951, 2042, 951, 2041, 953, 2039, 966, 2039, 967, 2038, 967, 2037, 969, 2035, 970, 2035, 971, 2034, 971, 2033, 972, 2032, 973, 2032, 975, 2030, 975, 2029, 973, 2027, 972, 2027, 971, 2026, 971, 2025, 970, 2024, 969, 2024, 967, 2022, 967, 2021, 966, 2020, 965, 2020, 963, 2018, 963, 2013, 962, 2012, 961, 2012, 959, 2010, 959, 2005, 958, 2004, 957, 2004, 955, 2002, 955, 1997, 954, 1996, 953, 1996, 951, 1994, 951, 1993, 950, 1992, 949, 1992, 947, 1990, 947, 1989, 946, 1988, 941, 1988, 939, 1986, 939, 1985, 938, 1984, 937, 1984, 935, 1982, 935, 1981, 934, 1980, 933, 1980, 931, 1978, 931, 1977, 930, 1976, 929, 1976, 927, 1974, 927, 1973, 926, 1972, 925, 1972, 923, 1970, 923, 1969, 922, 1968, 921, 1968, 919, 1966, 919, 1965, 918, 1964, 913, 1964, 911, 1962, 911, 1961, 910, 1960, 909, 1960, 907, 1958, 907, 1957, 906, 1956, 901, 1956, 899, 1954, 899, 1953, 898, 1952, 889, 1952, 887, 1950, 887, 1949, 886, 1948, 869, 1948, 867, 1946, 867, 1945, 866, 1944, 857, 1944, 855, 1942, 855, 1941, 854, 1940, 841, 1940, 839, 1938, 839, 1937, 838, 1936, 837, 1936, 835, 1934, 835, 1933, 834, 1932, 833, 1932, 831, 1930, 831, 1929, 830, 1928, 829, 1928, 827, 1926, 827, 1921, 826, 1920, 825, 1920, 823, 1918, 823, 1917, 822, 1916, 821, 1916, 819, 1914, 819, 1909, 818, 1908, 817, 1908, 815, 1906, 815, 1905, 814, 1904, 813, 1904, 811, 1902, 811, 1901, 810, 1900, 809, 1900, 807, 1898, 807, 1893, 806, 1892, 805, 1892, 803, 1890, 803, 1889, 802, 1888, 801, 1888, 799, 1886, 799, 1885, 798, 1884, 797, 1884, 795, 1882, 795, 1881, 794, 1880, 793, 1880, 791, 1878, 791, 1873, 790, 1872, 789, 1872, 787, 1870, 787, 1869, 786, 1868, 785, 1868, 783, 1866, 783, 1865, 782, 1864, 781, 1864, 779, 1862, 779, 1861, 778, 1860, 773, 1860, 771, 1858, 771, 1857, 770, 1856, 769, 1856, 767, 1854, 767, 1853, 766, 1852, 765, 1852, 763, 1850, 763, 1849, 762, 1848, 761, 1848, 759, 1846, 759, 1845, 758, 1844, 753, 1844, 751, 1842, 751, 1841, 750, 1840, 749, 1840, 747, 1838, 747, 1837, 746, 1836, 741, 1836, 739, 1834, 739, 1833, 738, 1832, 737, 1832, 735, 1830, 735, 1829, 734, 1828, 729, 1828, 727, 1826, 727, 1821, 726, 1820, 725, 1820, 723, 1818, 723, 1817, 722, 1816, 721, 1816, 719, 1814, 719, 1813, 718, 1812, 713, 1812, 711, 1810, 711, 1809, 710, 1808, 709, 1808, 707, 1806, 707, 1805, 706, 1804, 705, 1804, 703, 1802, 703, 1801, 702, 1800, 693, 1800, 691, 1798, 691, 1797, 690, 1796, 645, 1796, 643, 1794, 643, 1793, 642, 1792, 641, 1792, 639, 1790, 639, 1789, 638, 1788, 637, 1788, 635, 1786, 635, 1785, 634, 1784, 633, 1784, 631, 1782, 631, 1781, 630, 1780, 629, 1780, 627, 1778, 627, 1773, 626, 1772, 625, 1772, 623, 1770, 623, 1765, 622, 1764, 621, 1764, 619, 1762, 619, 1761, 618, 1760, 613, 1760, 611, 1758, 611, 1757, 610, 1756, 609, 1756, 607, 1754, 607, 1753, 606, 1752, 601, 1752, 599, 1750, 599, 1749, 598, 1748, 593, 1748, 591, 1746, 591, 1745, 590, 1744, 589, 1744, 587, 1742, 587, 1741, 586, 1740, 585, 1740, 583, 1738, 583, 1737, 582, 1736, 581, 1736, 579, 1734, 579, 1733, 578, 1732, 577, 1732, 575, 1730, 575, 1729, 574, 1728, 573, 1728, 571, 1726, 571, 1725, 570, 1724, 569, 1724, 567, 1722, 567, 1721, 566, 1720, 565, 1720, 563, 1718, 563, 1717, 562, 1716, 561, 1716, 559, 1714, 559, 1709, 558, 1708, 557, 1708, 555, 1706, 555, 1701, 554, 1700, 553, 1700, 551, 1698, 551, 1693, 550, 1692, 549, 1692, 547, 1690, 547, 1689, 546, 1688, 545, 1688, 543, 1686, 543, 1685, 542, 1684, 541, 1684, 539, 1682, 539, 1677, 538, 1676, 537, 1676, 535, 1674, 535, 1673, 534, 1672, 529, 1672, 527, 1670, 527, 1669, 526, 1668, 525, 1668, 523, 1666, 523, 1665, 522, 1664, 521, 1664, 519, 1662, 519, 1661, 518, 1660, 517, 1660, 515, 1658, 515, 1657, 514, 1656, 509, 1656, 507, 1654, 507, 1653, 506, 1652, 505, 1652, 503, 1650, 503, 1649, 502, 1648, 493, 1648, 491, 1646, 491, 1645, 490, 1644, 485, 1644, 483, 1642, 483, 1641, 482, 1640, 477, 1640, 475, 1638, 475, 1637, 474, 1636, 469, 1636, 467, 1634, 467, 1633, 466, 1632, 465, 1632, 463, 1630, 463, 1629, 462, 1628, 461, 1628, 459, 1626, 459, 1625, 458, 1624, 453, 1624, 451, 1622, 451, 1621, 450, 1620, 449, 1620, 447, 1618, 447, 1617, 446, 1616, 445, 1616, 443, 1614, 443, 1609, 442, 1608, 441, 1608, 439, 1606, 439, 1605, 438, 1604, 437, 1604, 435, 1602, 435, 1601, 434, 1600, 429, 1600, 427, 1598, 427, 1597, 426, 1596, 425, 1596, 423, 1594, 423, 1593, 422, 1592, 421, 1592, 419, 1590, 419, 1589, 418, 1588, 417, 1588, 415, 1586, 415, 1581, 414, 1580, 413, 1580, 411, 1578, 411, 1577, 410, 1576, 409, 1576, 407, 1574, 407, 1573, 406, 1572, 405, 1572, 403, 1570, 403, 1569, 402, 1568, 401, 1568, 399, 1566, 399, 1565, 398, 1564, 393, 1564, 391, 1562, 391, 1561, 390, 1560, 389, 1560, 387, 1558, 387, 1557, 386, 1556, 377, 1556, 375, 1554, 375, 1553, 374, 1552, 369, 1552, 367, 1550, 367, 1545, 366, 1544, 365, 1544, 363, 1542, 363, 1525, 362, 1524, 361, 1524, 359, 1522, 359, 1509, 358, 1508, 357, 1508, 355, 1506, 355, 1501, 354, 1500, 353, 1500, 351, 1498, 351, 1493, 353, 1491, 354, 1491, 355, 1490, 355, 1485, 356, 1484, 357, 1484, 359, 1482, 359, 1481, 358, 1480, 325, 1480, 323, 1478, 323, 1477, 322, 1476, 317, 1476, 315, 1474, 315, 1473, 314, 1472, 309, 1472, 307, 1470, 307, 1469, 306, 1468, 301, 1468, 299, 1466, 299, 1465, 298, 1464, 289, 1464, 287, 1462, 287, 1461, 286, 1460, 281, 1460, 279, 1458, 279, 1457, 278, 1456, 277, 1456, 275, 1454, 275, 1453, 274, 1452, 269, 1452, 267, 1450, 267, 1449, 266, 1448, 265, 1448, 263, 1446, 263, 1445, 262, 1444, 261, 1444, 259, 1442, 259, 1441, 258, 1440, 257, 1440, 255, 1438, 255, 1437, 254, 1436, 253, 1436, 251, 1434, 251, 1433, 250, 1432, 249, 1432, 247, 1430, 247, 1429, 246, 1428, 245, 1428, 243, 1426, 243, 1425, 242, 1424, 241, 1424, 239, 1422, 239, 1421, 238, 1420, 237, 1420, 235, 1418, 235, 1417, 234, 1416, 229, 1416, 227, 1414, 227, 1413, 226, 1412, 225, 1412, 223, 1410, 223, 1405, 222, 1404, 221, 1404, 219, 1402, 219, 1401, 218, 1400, 217, 1400, 215, 1398, 215, 1393, 214, 1392, 213, 1392, 211, 1390, 211, 1385, 210, 1384, 209, 1384, 207, 1382, 207, 1377, 206, 1376, 205, 1376, 203, 1374, 203, 1373, 202, 1372, 201, 1372, 199, 1370, 199, 1369, 198, 1368, 197, 1368, 195, 1366, 195, 1365, 194, 1364, 189, 1364, 187, 1362, 187, 1361, 186, 1360, 185, 1360, 183, 1358, 183, 1357, 182, 1356, 177, 1356, 175, 1354, 175, 1353, 174, 1352, 173, 1352, 171, 1350, 171, 1349, 170, 1348, 169, 1348, 167, 1346, 167, 1345, 166, 1344, 165, 1344, 163, 1342, 163, 1341, 162, 1340, 161, 1340, 159, 1338, 159, 1337, 158, 1336, 157, 1336, 155, 1334, 155, 1333, 154, 1332, 153, 1332, 151, 1330, 151, 1329, 150, 1328, 149, 1328, 147, 1326, 147, 1325, 146, 1324, 145, 1324, 143, 1322, 143, 1321, 142, 1320, 141, 1320, 139, 1318, 139, 1317, 138, 1316, 137, 1316, 135, 1314, 135, 1313, 134, 1312, 133, 1312, 131, 1310, 131, 1309, 130, 1308, 125, 1308, 123, 1306, 123, 1305, 122, 1304, 121, 1304, 119, 1302, 119, 1301, 118, 1300, 93, 1300, 91, 1298, 91, 1297, 90, 1296, 85, 1296, 83, 1294, 83, 1293, 82, 1292, 81, 1292, 79, 1290, 79, 1289, 78, 1288, 77, 1288, 75, 1286, 75, 1285, 74, 1284, 73, 1284, 71, 1282, 71, 1281, 70, 1280, 69, 1280, 67, 1278, 67, 1277, 66, 1276, 65, 1276, 63, 1274, 63, 1269, 62, 1268, 61, 1268, 59, 1266, 59, 1265, 58, 1264, 57, 1264, 55, 1262, 55, 1233, 57, 1231, 58, 1231, 59, 1230, 59, 1205, 58, 1204, 57, 1204, 55, 1202, 55, 1197, 54, 1196, 53, 1196, 51, 1194, 51, 1193, 50, 1192, 45, 1192, 43, 1190, 43, 1189, 42, 1188, 37, 1188, 35, 1186, 35, 1185, 34, 1184], "class": 1, "name": "inference"}, "type": "Feature"}], "type": "FeatureCollection"}'
    # driver = ogr.GetDriverByName('GeoJSON')
    # # input SpatialReference
    # inSpatialRef = osr.SpatialReference()
    # inSpatialRef.ImportFromEPSG(5186)
    # # output SpatialReference
    # outSpatialRef = osr.SpatialReference()
    # outSpatialRef.ImportFromEPSG(3857)
    # # create the CoordinateTransformation
    # coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # # get the input layer
    # inDataSet = gdal.OpenEx(str(geojson_metadata))
    # inLayer = inDataSet.GetLayer()
    # # create the output layer
    # outputShapefile = "test3.geojson"
    # if os.path.exists(outputShapefile):
    #     driver.DeleteDataSource(outputShapefile)
    # outDataSet = driver.CreateDataSource(outputShapefile)
    # outLayer = outDataSet.CreateLayer("", geom_type=ogr.wkbMultiPolygon, srs=outSpatialRef)
    # # add fields
    # inLayerDefn = inLayer.GetLayerDefn()
    # for i in range(0, inLayerDefn.GetFieldCount()):
    #     fieldDefn = inLayerDefn.GetFieldDefn(i)
    #     outLayer.CreateField(fieldDefn)
    # # get the output layer's feature definition
    # outLayerDefn = outLayer.GetLayerDefn()
    # # loop through the input features
    # inFeature = inLayer.GetNextFeature()
    # while inFeature:
    #     # get the input geometry
    #     geom = inFeature.GetGeometryRef()
    #     # reproject the geometry
    #     geom.Transform(coordTrans)
    #     # create a new feature
    #     outFeature = ogr.Feature(outLayerDefn)
    #     # set the geometry and attribute
    #     outFeature.SetGeometry(geom)
    #     for i in range(0, outLayerDefn.GetFieldCount()):
    #         outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), str(inFeature.GetField(i)))
    #     # add the feature to the shapefile
    #     outLayer.CreateFeature(outFeature)
    #     # dereference the features and get the next input feature
    #     outFeature = None
    #     inFeature = inLayer.GetNextFeature()
    # # Save and close the shapefiles
    # inDataSet = None
    # outDataSet = None

