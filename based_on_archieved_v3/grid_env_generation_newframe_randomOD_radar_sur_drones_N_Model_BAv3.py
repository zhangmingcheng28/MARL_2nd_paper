# -*- coding: utf-8 -*-
"""
@Time    : 8/12/2022 10:01 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as matPolygon
import math
from shapely.strtree import STRtree
from shapely import affinity
from shapely.geometry import Polygon, LineString
from shapely import GeometryCollection
from scipy import ndimage
from shapely.geometry.point import Point
import geopandas as gpd
import shapely
import random
import statistics


def coordinate_to_meter(target, max_, min_, span):
    portion = max_ - min_
    meter_per_unit = portion / span  # this is the length in meter represented by each unit of input
    meter = (target - min_) * meter_per_unit
    return meter  # conversion of the targeted coordinate into meter


def create_random_circle(minX, maxX, minY, maxY):
    cenX = random.randint(minX, maxX)
    cenY = random.randint(minY, maxY)
    p_ = Point(cenX, cenY)
    randomCricle = p_.buffer(50.0)  # this buffer is radius
    return randomCricle


def generate_circle(cx,cy):
    p_ = Point(cx, cy)
    circle_ = p_.buffer(2.5)  # this buffer is radius
    return circle_


def shapelypoly_to_matpoly(shapelyPolgon, inFill=False, inEdgecolor='black'):
    xcoo, ycoo = shapelyPolgon.exterior.coords.xy
    matPolyConverted = matPolygon(xy=list(zip(xcoo, ycoo)), fill=inFill, edgecolor=inEdgecolor)
    return matPolyConverted


def initialize_3d_array_environment(girdLength, maxX, maxY, maxZ):  # grid is a cube
    arrlength_x = math.ceil(maxX / girdLength)
    arrlength_y = math.ceil(maxY / girdLength)
    arrlength_z = math.ceil(maxZ / girdLength)
    initialized3DArray = np.zeros((arrlength_x, arrlength_y, arrlength_z))
    return initialized3DArray


def square_grid_intersection(strTreePolyset, gridToTest, buildingPolygonDict):
    occupied = 0
    height = 0
    polygons_in_vicinity_index = strTreePolyset.query(gridToTest)  # will return possible polygons around the tested grids, including the tested grid itself. Be careful of double counting!
    if len(polygons_in_vicinity_index) == 0:
        return occupied, height

    if len(polygons_in_vicinity_index) == 1:
        possiblePoly = strTreePolyset.geometries.take(polygons_in_vicinity_index).tolist()  # this is shapely polygon
        matp_PolyConvert = shapelypoly_to_matpoly(possiblePoly[0])
        matp_gridToTest = shapelypoly_to_matpoly(gridToTest, True)


        # matplotlib.use('Qt5Agg')
        # fig, ax = plt.subplots(1, 1)
        # # Add the polygon to the axis
        # ax.add_patch(matPolyConvert)
        # ax.add_patch(gridToTest)
        # plt.autoscale()
        # # Display the plot
        # plt.show()

        if possiblePoly[0].disjoint(gridToTest):  # one possible polygon around, but does not intersect or equal
            pass
        else:
            occupied = 1
            height = buildingPolygonDict[id(possiblePoly[0])]
    else:  # if current gridToTest have spatial relationship wih two or more building polygons
        heightToAverage = []
        for possiblePoly_idx in polygons_in_vicinity_index:
            possiblePoly = strTreePolyset.geometries.take(possiblePoly_idx)  # this is shapely polygon
            if possiblePoly.disjoint(gridToTest):
                pass  # disjoint, no action required
            else:
                occupied = 1
                heightToAverage.append(buildingPolygonDict[id(possiblePoly)])
        # after look through two possible polygons and no spatial relationship between the gridToTest we can just return the result
        if occupied == 0:
            return occupied, height
        if len(heightToAverage) > 0:
            height = statistics.mean(heightToAverage)
        else:  # "heightToAverage" only has a single item
            height = heightToAverage[0]
    return occupied, height


def env_generation(shapeFilePath, bound):  # input: string of file path, output: STRtree object of polygons
    shape = gpd.read_file(shapeFilePath)
    # check for duplicates and remove it
    ps = pd.DataFrame(shape)
    ps["geometry"] = ps["geometry"].apply(lambda geom: geom.wkb)  # convert to wkb, which is hashable, or else cannot apply drop.duplicates()
    ps = ps.drop_duplicates(["geometry"])  # apply drop_duplicates() function
    ps["geometry"] = ps["geometry"].apply(lambda geom: shapely.wkb.loads(geom))  # convert back to shaply polygon
    # End of remove duplicates
    # convert coordinate to meters, both x and y start from 0
    polySet_buildings = []
    maxHeight = 0
    polyDict = {}
    for index, row in ps.iterrows():  # "ps" already dropped the duplicates, but the index is unchange from the "shape"
        currentPolyHeight = row[2]
        if currentPolyHeight >= maxHeight:
            maxHeight = currentPolyHeight
        coordsToChange = row[6].exterior.coords[:]
        for pos, item in enumerate(coordsToChange):
            # these values are specifically for the individual environment, generated by using SVY21
            x_meter = coordinate_to_meter(item[0], 16262.89690000005, 14550, 1800)
            y_meter = coordinate_to_meter(item[1], 37448.60029999912, 36200, 1300)
            coordsToChange[pos] = (x_meter, y_meter)
        poly_transformed = Polygon(coordsToChange)
        ps.at[index, 'geometry'] = poly_transformed
        polyDict[id(poly_transformed)] = row[2]  # shapely.Polygon itself is not hashable, but id(Polygon) is hashable
        polySet_buildings.append(poly_transformed)  # this is the polygon in terms of meters
    # populate STRtree
    tree_of_polySet_buildings = STRtree(polySet_buildings)

    # generate 3D array of 0 and 1 based on the dictionary of polygons with height and grid intersection
    maxX = 1800
    maxY = 1300
    gridLength = 10
    envMatrix = initialize_3d_array_environment(gridLength, maxX, maxY, math.ceil(maxHeight))
    x_lower = math.ceil(bound[0] / gridLength)
    x_higher = math.ceil(bound[1] / gridLength)
    y_lower = math.ceil(bound[2] / gridLength)
    y_higher = math.ceil(bound[3] / gridLength)

    gridPoly_beforeFill = []
    for xi in range(envMatrix.shape[0]):
        for yj in range(envMatrix.shape[1]):
            gridPoint = Point(xi*gridLength, yj*gridLength)
            gridPointPoly = gridPoint.buffer(gridLength / 2, cap_style=3)  # cap_style=3 for grid size bound
            # if gridPointPoly.equals(Polygon(([25, 805], [25, 795], [15, 795], [15, 805], [25, 805]))):
            #     print('debug')
            occpied_avgHeigh = square_grid_intersection(tree_of_polySet_buildings, gridPointPoly, polyDict)
            if occpied_avgHeigh[0]:
                matrixHeight = math.ceil(occpied_avgHeigh[1]/gridLength)  # NOTE: for env matrix, height is also scaled accroding to grid length
                envMatrix[xi, yj, 0:matrixHeight] = 1
                gridPoly_beforeFill.append(gridPointPoly)
                # for display occupied grids
                #ax.add_patch(PolygonPatch(gridPointPoly))


    # After the prelimary world map has been build, we need to cover-up the holes that is surrounded by the occupied grids
    env_map = ndimage.binary_fill_holes(envMatrix[:, :, 0])  # env_layer fill holes
    env_map_bounded = env_map[x_lower:x_higher, y_lower:y_higher]
    gridPoly_ones = []
    gridPoly_zero = []
    outPoly = []
    for ix in range(env_map.shape[0]):
        for iy in range(env_map.shape[1]):
            if (ix * gridLength <= bound[1]) and (ix * gridLength >= bound[0]) and (iy * gridLength <= bound[3]) and (
                    iy * gridLength >= bound[2]):
                grid_point_toTest = Point(ix * gridLength, iy * gridLength)
                grid_poly_toTest = grid_point_toTest.buffer(gridLength / 2, cap_style=3)
                if env_map[ix][iy] == 1:  # get the grid index that is occupied, meaning yield 1 in the index position
                    # transform these grid index information to the actual size map
                    gridPoly_ones.append(grid_poly_toTest)
                else:
                    gridPoly_zero.append(grid_poly_toTest)
    outPoly.append([gridPoly_ones, gridPoly_zero])

        # display all building polygon
    # for poly in polySet:
    #     ax.add_patch(PolygonPatch(poly))
    return env_map_bounded, polySet_buildings, gridLength, outPoly, (maxX, maxY)  # return an occupied 3D array

def pointgen(p1, p2, p3, totalnum):
    xySet = []
    for _ in range(totalnum):
        alpha = np.random.uniform(0, 1, 1)
        d = 1-alpha
        beta = np.random.uniform(0, 1, 1)
        d_beta = d*beta
        gamma = 1-(alpha + d_beta)
        #print(alpha + d_beta + gamma)
        x = (alpha*p1.x) + (d_beta*p2.x) + (gamma*p3.x)
        y = (alpha*p1.y) + (d_beta*p2.y) + (gamma*p3.y)
        xySet.append([x, y])
    return xySet

def polygon_random_points (poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < num_points:
            random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
            if (random_point.within(poly)):
                points.append(random_point)
    return points



# for running test ------------------------------------------------------------------
if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    fig, ax = plt.subplots(1, 1)
    shape = gpd.read_file('F:\githubClone\deep_Q_learning\lakesideMap\lakeSide.shp')
    # check for duplicates and remove it
    ps = pd.DataFrame(shape)
    ps["geometry"] = ps["geometry"].apply(lambda geom: geom.wkb)  # convert to wkb, which is hashable, or else cannot apply drop.duplicates()
    ps = ps.drop_duplicates(["geometry"])  # apply drop_duplicates() function
    ps["geometry"] = ps["geometry"].apply(lambda geom: shapely.wkb.loads(geom))  # convert back to shaply polygon
    # End of remove duplicates
    #print('end')
    # convert coordinate to meters, both x and y start from 0
    polySet = []
    maxHeight = 0
    polyDict = {}
    for index, row in ps.iterrows():  # ps already dropped the duplicates, but the index is unchange from the "shape"
        currentPolyHeight = row[2]
        if currentPolyHeight >= maxHeight:
            maxHeight = currentPolyHeight
        coordsToChange = row[6].exterior.coords[:]
        for pos, item in enumerate(coordsToChange):
            x_meter = coordinate_to_meter(item[0], 16262.89690000005, 14550, 1800)
            y_meter = coordinate_to_meter(item[1], 37448.60029999912, 36200, 1300)
            coordsToChange[pos] = (x_meter, y_meter)
        poly_transformed = Polygon(coordsToChange)
        ps.at[index, 'geometry'] = poly_transformed
        polyDict[id(poly_transformed)] = row[2]  # shapely.Polygon itself is not hashable, but id(Polygon) is hashable
        polySet.append(poly_transformed)  # this is the polygon in terms of meters
    # end of convert to meters
    # populate polygon into STRtree
    tree_of_polySet = STRtree(polySet)

    # generate 3D array of 0 and 1 based on the dictionary of polygons with height and grid intersection
    maxX = 1800
    maxY = 1300
    gridLength = 10
    envMatrix = initialize_3d_array_environment(gridLength, maxX, maxY, math.ceil(maxHeight))
    for xi in range(envMatrix.shape[0]):
        for yj in range(envMatrix.shape[1]):
            gridPoint = Point(xi*gridLength, yj*gridLength)
            gridPointPoly = gridPoint.buffer(gridLength / 2, cap_style=3)  # cap_style=3 for grid size bound
            # if gridPointPoly.equals(Polygon(([25, 805], [25, 795], [15, 795], [15, 805], [25, 805]))):
            #     print('debug')
            occpied_avgHeigh = square_grid_intersection(tree_of_polySet, gridPointPoly, polyDict)
            if occpied_avgHeigh[0]:
                matrixHeight = math.ceil(occpied_avgHeigh[1]/gridLength)
                envMatrix[xi, yj, 0:matrixHeight] = 1
                # for display occupied grids
                matp_gridPointPoly = shapelypoly_to_matpoly(gridPointPoly, True, 'grey')  # the 3rd parameter is the edge color
                #ax.add_patch(matp_gridPointPoly)


    # After the prelimary world map has been build, we need to cover-up the holes that is surrounded by the occupied grids
    env_layer = ndimage.binary_fill_holes(envMatrix[:, :, 0])  # env_layer fill holes
    polySet_filled = []
    for ix in range(env_layer.shape[0]):
        for iy in range(env_layer.shape[1]):
            if env_layer[ix][iy] == 1:  # get the grid index that is occupied, meaning yield 1 in the index position
                # transform these grid index information to the actual size map
                grid_point = Point(ix * gridLength, iy * gridLength)
                grid_poly = grid_point.buffer(gridLength / 2, cap_style=3)
                mat_grid_poly = shapelypoly_to_matpoly(grid_poly, True, 'red')
                ax.add_patch(mat_grid_poly)
                polySet_filled.append(grid_poly)



    #display all building polygon
    for poly in polySet:
        matp_poly = shapelypoly_to_matpoly(poly, True, 'black')  # the 3rd parameter is the edge color
        #ax.add_patch(matp_poly)

    # -----------to prove the env-grid generated is exactly the same compared to the original shape---------- #
    #access any level:  # take note: only at level 0 the top view is the same as the original env. The building has height, so at higher level, the top view may not be the same as the original environemnt in the top view.
    # due to the reason above, so when used for environment simulation, we only use level = 0
    level = 0
    sc_x = []
    sc_y = []
    sc_cx = []
    sc_cy = []
    layer = envMatrix[:, :, level]
    for ix in range(layer.shape[0]):
        for iy in range(layer.shape[1]):
            if layer[ix][iy] == 1:
                gp = Point(ix, iy)
                sc_x.append(ix)
                sc_y.append(iy)
                gppoly = gp.buffer(1/2, cap_style=3)
                #ax.add_patch(PolygonPatch(gppoly))
            if (47 <= ix < 53) and (32 <= iy < 38) and (layer[ix][iy] == 1):
                c_gp = Point(ix, iy)
                sc_cx.append(ix)
                sc_cy.append(iy)
                gppoly = c_gp.buffer(1/2, cap_style=3)
                #ax.add_patch(PolygonPatch(gppoly, fc='red'))
            if ix == 50 and iy == 35:
                p = Point(ix, iy)
                gppoly_p = p.buffer(1 / 2, cap_style=3)
                #ax.add_patch(PolygonPatch(gppoly_p, fc='green'))
    p1 = Point(480, 305)
    p2 = Point(520, 280)
    origin_cir = p1.buffer(5, cap_style=1)
    line1 = LineString([p1, p2])
    line_rot_center = affinity.rotate(line1, 90, 'center')  # this rotation is done in CCW. Meaning, if rotate 0-deg, the p1 is actually the p3.
    test_line = affinity.scale(line_rot_center, xfact=0.5, yfact=0.5, zfact=1.0, origin='center')
    p3 = Point(line_rot_center.xy[0][0], line_rot_center.xy[1][0])
    p4 = Point(line_rot_center.xy[0][1], line_rot_center.xy[1][1])
    testPoly = Polygon([p1, p3, p2])
    #ax.add_patch(PolygonPatch(testPoly, fc='y', alpha=0.1))
    #print(p1.distance(p2))
    p3_deltax = p1.distance(p2)*math.cos(math.radians(45))
    #print(math.cos(math.radians(45)))
    p4_deltay = p3_deltax
    #p3 = Point(p1.x + p3_deltax, p1.y)
    #p4 = Point(p1.x, p1.y + p4_deltay)
    poly = Polygon([[p1.x, p1.y], [p4.x, p4.y], [p2.x, p2.y], [p3.x, p3.y]])  # Polygon for polygon must input points in a CW manner, and no need to repeat the 1st point
    poly_1 = Polygon([[p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]])
    alongPoly_1 = Polygon([[p1.x, p1.y], [p4.x, p4.y], [p3.x, p3.y]])
    # plt.plot(*line1.xy)
    # plt.plot(*test_line.xy)
    # plt.plot(p1.x, p1.y, 'bo')
    # plt.plot(p2.x, p2.y, 'bo')
    # plt.plot(p3.x, p3.y, 'go')
    # plt.plot(p4.x, p4.y, 'ro')
    existPoly = origin_cir.difference(alongPoly_1)
    existPoly = poly_1.difference(origin_cir)
    existPoly = existPoly.difference(p2.buffer(10, cap_style=1))
    ptList = pointgen(p1, p2, p3, 20)
    tp1 = Point(530.3161859583022, 288.3363714879051)
    tpd = Point(529.0625514902796, 292.3361393585554)
    new_tpt = Point(578.8567973363544, 295.58253057010916)
    # plt.plot(tp1.x, tp1.y, 'bo')
    # plt.plot(tpd.x, tpd.y, 'go')
    # plt.plot(new_tpt.x, new_tpt.y, 'ro')
    #plt.plot(p4.x, p4.y, 'ro')
    #ax.add_patch(PolygonPatch(existPoly, fc='y', alpha=0.3))
    #ax.add_patch(PolygonPatch(existPoly))
    # ax.add_patch(PolygonPatch(origin_cir))
    # ax.add_patch(PolygonPatch(alongPoly_1, alpha=0.1))
    #ax.add_patch(PolygonPatch(existPoly, fc='k'))
    # a = Point(1, 1).buffer(1.5)
    # b = Point(2, 1).buffer(1.5)
    # patch1 = PolygonPatch(a, alpha=0.2, zorder=1)
    # ax.add_patch(patch1)
    # patch2 = PolygonPatch(b, alpha=0.2, zorder=1)
    # ax.add_patch(patch2)
    # c = a.difference(b)
    # patchc = PolygonPatch(c, alpha=0.5, zorder=2)
    # ax.add_patch(patchc)
    # ax.add_patch(PolygonPatch(p1, fc='y'))
    # ax.add_patch(PolygonPatch(p2, fc='y'))
    # ax.add_patch(PolygonPatch(p3, fc='y'))
    # ax.add_patch(PolygonPatch(p4, fc='y'))
    #ax.add_patch(p1)
    #ax.add_patch(PolygonPatch(p1, fc='y'))
    #print(poly_1.minimum_clearance)
    points = polygon_random_points(testPoly, 20)
    # observable_env_2dArray = self.staticOBS_STRarr[
    #                          cur_pos_mapped[0] - seen_grid_halfrange: cur_pos_mapped[0] + seen_grid_halfrange,
    #                          cur_pos_mapped[1] - seen_grid_halfrange: cur_pos_mapped[1] + seen_grid_halfrange,
    #                          self.globalAltitude]



    # cut_layer = layer[47:53, 32:38]
    # for ix in range(cut_layer.shape[0]):
    #     for iy in range(cut_layer.shape[1]):
    #         if cut_layer[ix][iy] == 1:
    #             gp = Point(ix, iy)
    #             sc_cx.append(ix)
    #             sc_cy.append(iy)
    #             gppoly = gp.buffer(1/2, cap_style=3)
    #             ax.add_patch(PolygonPatch(gppoly, ))
    #
    # #plt.scatter(sc_x, sc_y, c='red')
    # plt.scatter(sc_cx, sc_cy, c='red')
    # ------------------------------------------------------------------------------------------------------ #

    #create random circle
    randCircle = create_random_circle(0, 1800, 0, 1300)
    result = []
    # for ip in ptList:
    #     if Point(ip).within(existPoly):
    #         plt.plot(ip[0], ip[1], 'bo')
        #plt.plot(ip.x, ip.y, 'bo')

    # cir1 = generate_circle(500, 350)
    # ax.add_patch(PolygonPatch(cir1, fc='red'))
    # cir2 = generate_circle(560, 325)
    # ax.add_patch(PolygonPatch(cir2, fc='red'))
    # cir3 = generate_circle(650, 280)
    # ax.add_patch(PolygonPatch(cir3, fc='red'))
    # cir4 = generate_circle(600, 305)  # one possible starting location for intruder
    # ax.add_patch(PolygonPatch(cir4, fc='yellow'))
    # cir5 = generate_circle(560, 350)  # another possible starting location for intruder
    # ax.add_patch(PolygonPatch(cir5, fc='yellow'))
    # cir1 = generate_circle(500, 350)
    # ax.add_patch(PolygonPatch(cir1))
    # while len(result)==0:
    #     result = tree_of_polySet.query(randCircle)
    #     if len(result) != 0:
    #         ax.add_patch(PolygonPatch(randCircle))
    #         for poly in result:
    #             ax.add_patch(PolygonPatch(poly))
    #     randCircle = create_random_circle(0, 1800, 0, 1300)
    # print(len(result))
    #ax.set(xlim=(20, 200), ylim=(600, 1000))
    #ax.set(xlim=(0, 1800), ylim=(0, 1300))
    #ax.set(xlim=(0, 180), ylim=(0, 130))
    # plt.plot(450, 340, marker='s', color='r')
    # plt.plot(540, 355, marker='s', color='r')
    # plt.plot(595, 325, marker='s', color='r')
    ax.set_aspect('equal')
    plt.axis('equal')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("N-S direction", fontsize=14)
    plt.ylabel("E-W direction", fontsize=14)
    plt.show()
# END for running test ------------------------------------------------------------------

