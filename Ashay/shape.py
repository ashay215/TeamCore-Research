import shapefile as shp
import matplotlib.pyplot as plt

# import geopandas as gpd
# matplotlib.use('TkAgg')   

# shape=gpd.read_file('roads_qenp/qenp_roadsWGS84.shp')
# shape.plot()

# from descartes import PolygonPatch
# import shapefile
# sf=shapefile.Reader('roads_qenp/qenp_roadsWGS84.shp')
# poly=sf.shape(1).__geo_interface__
# fig = plt.figure() 
# ax = fig.gca() 
# ax.add_patch(PolygonPatch(poly, fc='#ffffff', ec='#000000', alpha=0.5, zorder=2 ))
# ax.axis('scaled')
# plt.show()


sf = shp.Reader("Water/mfca_water.shp")

plt.figure()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)
plt.show()